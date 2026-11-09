using System.Collections.Concurrent;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace Bridge;

/// <summary>One writer lock, one reader, and request-ID based out-of-order replies.</summary>
internal sealed class JsonLineTransport : IDisposable
{
    private readonly StreamWriter _writer;
    private readonly StreamReader _reader;
    private readonly object _writeLock = new();
    private readonly ConcurrentDictionary<string, PendingResponse> _pending = new();
    private readonly Action? _onFatalError;
    private readonly Task _readTask;
    private Exception? _failure;
    private bool _disposed;

    public JsonLineTransport(StreamWriter writer, StreamReader reader, Action? onFatalError = null)
    {
        _writer = writer;
        _reader = reader;
        _onFatalError = onFatalError;
        _readTask = Task.Factory.StartNew(ReadResponses, CancellationToken.None,
            TaskCreationOptions.LongRunning, TaskScheduler.Default);
    }

    public string Send(string json, int timeoutMilliseconds, IDisposable? requestLifetime = null)
    {
        if (timeoutMilliseconds <= 0)
            throw new ArgumentOutOfRangeException(nameof(timeoutMilliseconds));
        string id = Guid.NewGuid().ToString("N");
        var pending = new PendingResponse(requestLifetime);
        // Request objects are JSON objects produced by JsonSerializer, not arbitrary strings.
        string message = "{\"request_id\":\"" + id + "\"," + json.Substring(1);
        lock (_writeLock)
        {
            if (_disposed || _failure != null)
            {
                pending.DisposeLifetime();
                throw new IOException("Python response channel is closed.", _failure);
            }
            if (_pending.Count >= 26)
            {
                pending.DisposeLifetime();
                throw new InvalidOperationException("Too many pending/late Python requests.");
            }
            if (!_pending.TryAdd(id, pending))
                throw new InvalidOperationException("Duplicate request ID.");
            try
            {
                _writer.WriteLine(message);
                _writer.Flush();
            }
            catch
            {
                if (_pending.TryRemove(id, out var removed)) removed.DisposeLifetime();
                throw;
            }
        }
        try
        {
            if (Task.WaitAny(new Task[] { pending.Completion.Task }, timeoutMilliseconds) < 0)
            {
                if (pending.Completion.Task.IsCompleted)
                    return pending.Completion.Task.GetAwaiter().GetResult();
                var timeout = new TimeoutException(
                    $"Python request {id} timed out after {timeoutMilliseconds} ms; the worker was terminated.");
                try { _onFatalError?.Invoke(); } finally { FailAll(timeout); }
                throw timeout;
            }
            return pending.Completion.Task.GetAwaiter().GetResult();
        }
        catch
        {
            // A timeout does not cancel Python. Keep the mapping alive until its
            // matching reply arrives (or the worker dies); never reuse the ID.
            throw;
        }
    }

    private void ReadResponses()
    {
        try
        {
            while (true)
            {
                string? line = _reader.ReadLine();
                if (line == null)
                    throw new EndOfStreamException("Python closed stdout.");
                using JsonDocument response = JsonDocument.Parse(line);
                if (!response.RootElement.TryGetProperty("request_id", out JsonElement idElement)
                    || idElement.ValueKind != JsonValueKind.String)
                    throw new InvalidDataException("Python server must echo request_id. Deploy the updated server with this DLL.");
                string id = idElement.GetString()!;
                if (_pending.TryRemove(id, out var pending))
                {
                    pending.DisposeLifetime();
                    pending.Completion.TrySetResult(line);
                }
            }
        }
        catch (Exception exception)
        {
            try { _onFatalError?.Invoke(); } catch { }
            FailAll(exception);
        }
    }

    private void FailAll(Exception exception)
    {
        lock (_writeLock)
        {
            _failure = exception;
            foreach (var item in _pending)
                if (_pending.TryRemove(item.Key, out var completion))
                {
                    completion.DisposeLifetime();
                    completion.Completion.TrySetException(exception);
                }
        }
    }

    public void Dispose()
    {
        lock (_writeLock) { _disposed = true; }
        FailAll(new ObjectDisposedException(nameof(JsonLineTransport)));
        // PythonBridge owns and closes the streams/process, which unblocks ReadLine.
    }

    private sealed class PendingResponse
    {
        private IDisposable? _lifetime;
        public PendingResponse(IDisposable? lifetime)
        {
            _lifetime = lifetime;
            Completion = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        }
        public TaskCompletionSource<string> Completion { get; }
        public void DisposeLifetime() => Interlocked.Exchange(ref _lifetime, null)?.Dispose();
    }
}
