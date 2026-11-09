# SmartRay 16位深度Pitch矫正DLL

产物：`bin/Release/SmartRay.PitchCorrection.dll`。

C#托管类库，供当前.NET Framework 4.8 SmartRay Demo直接添加引用使用；不是通过`DllImport`调用的原生C接口。运行不依赖Python、NumPy、SmartRay SDK或额外NuGet包。该目录独立，不修改现有检测程序或外部相机Demo。

## C#调用

在上位机项目“引用 → 添加引用 → 浏览”选择DLL，将DLL随EXE部署。创建一个矫正器反复使用，它会在构造时保存参数并预计算65536个深度编码的映射表。修改参数后创建新实例。

```csharp
using SmartRay.PitchCorrection;

// 6度和0.02仅是调用示例，请换成实际标定值。
var corrector = new PitchCorrector(new PitchOptions
{
    Enabled = true,
    AngleDegrees = 6.0,
    ScanPitchMm = 0.02,
    ZStepMm = 0.00105,
    GrayOffset = 32768.0,
    ZOriginMm = 0.0,
    Axis = ScanAxis.Rows,
    Direction = TransformDirection.RecoverIdeal,
    ProjectionWorkers = 4
});

// data为完整采集完成事件传来的SensorImageData，不要使用显示拉伸图。
ushort[] input = data.ZMapImage.GetImage();
int width = data.Width;
int height = data.CurrentHeight;
ushort[] corrected = corrector.Correct(input, width, height);
// corrected.Length == width * height，逐行排列，原input不变。
```

复用上位机自己的输出缓冲：

```csharp
ushort[] output = new ushort[width * height];
corrector.CorrectInto(input, output, width, height);
// 后续同尺寸帧可以继续使用output，函数会清空并重新生成结果。
// 输入输出不能是同一数组。
```

`GetImage()`仍会拼接复制SDK数据包；DLL本身不读写PNG，也不负责相机采集、文件保存、共享内存或Python请求。结果可以直接写入共享内存；调用检测端时须确保入口支持raw uint16及对应宽高、字节序和行布局，不能只修改data_format。显示与留档可以独立处理。

## 参数

|参数|意义及范围|
|---|---|
|Enabled|默认true；false时逐值复制输入，包括0和65535，不执行补偿|
|AngleDegrees|带正负号的Pitch角，单位度，绝对值必须小于60|
|ScanPitchMm|扫描方向每行/列的真实步距，mm/profile，必须大于0；内触发时由调用方按扫描速度/剖面频率算出|
|ZStepMm|深度编码每增加1对应的毫米值，默认0.00105，必须大于0|
|GrayOffset|深度零点编码，默认32768|
|ZOriginMm|在深度解码后叠加的高度原点，mm，默认0|
|Axis|Rows默认：扫描沿图像行，改变行号；Columns：扫描沿列，改变列号|
|Direction|RecoverIdeal默认：测量值还原理想值；DocumentForward：文档正向映射，用于对照|
|ProjectionWorkers|并行条带数上限，默认1，必须为正整数；实际不超过非扫描轴尺寸，不设置CPU亲和性|

调用方传入的数组必须为紧密排列的单通道`ushort[width*height]`，索引为`row*width+col`。若从Bitmap读取含padding的缓冲，需先移除行尾填充。只接收实际采满的整幅数据；Demo的GetLastImageData可能返回部分图像，提交前要确认完整性。

同一个矫正器可以并发处理不同帧，输入在调用期间不得被外部修改；每次调用必须使用独立的输出数组。DLL内部参数与查找表只读，不缓存帧数据。

## 与现有算法的对应关系

设输入编码为d，`z=(d-GrayOffset)*ZStepMm+ZOriginMm`，扫描坐标为q：

还原模式：

```text
q_out = round_to_even(q - z*tan(beta)/ScanPitchMm)
z_out = z/cos(beta)
d_out = round_to_even((z_out-ZOriginMm)/ZStepMm + GrayOffset)
```

文档正向模式使用`q_out=round_to_even(q+z*sin(beta)/ScanPitchMm)`和`z_out=z*cos(beta)`。

- 输入0、65535无效；输出有效编码范围1至65534。
- 最近邻正向投影，取整规则与NumPy rint一致：中点取偶数。先将连续偏移加到像素坐标，再取整。
- 投影坐标越界或输出编码超范围则丢弃，未填充位置为0，不插值补洞。
- 多点落到同一输出像素时，保留最大深度编码。
- 输出宽高与输入相同，按原画幅裁切；输入始终不被修改。
- 沿非扫描轴分配条带，每个任务独占其输出，避免并行散射写入竞争。

输入须为尚未补偿的原始深度。若SDK已做Pitch矫正，不应再执行本DLL补偿；若先用本DLL矫正再送入Python检测，也应关闭Python侧Pitch补偿，避免重复处理。高度标定参数必须与测高配置一致。

## 构建

Windows自带.NET Framework C#编译器即可，无需安装包或联网恢复依赖：

```powershell
& .\smartray_pitch_dll\build.ps1
```

也提供.NET Framework 4.8项目`SmartRay.PitchCorrection.csproj`，可在已安装对应开发包的Visual Studio中构建。

## 合成验证

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B .\smartray_pitch_dll\verification\verify.py
```

该验证编译一个C#控制台程序调用实际DLL，并与当前`python/algorithm/tilt_correction.py`输出逐像素对照。覆盖行/列、正负角度、两种方向、所有65536种深度编码、无效点、越界/碰撞、半像素取整、非默认原点、单行/列、多线程及禁用旁路；检查输入不变、复用缓冲清零、同实例跨帧并发和16种非法输入。未连接相机、未调用模型、未做真实图像性能测试。
