"""Gate tiny shared-array copies to expose lock blocking without a benchmark."""
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from algorithm.registration_process import SharedBoardInputs


class SharedPublicationTests(unittest.TestCase):
    def owner(self):
        owner=SharedBoardInputs();owner.acquire()
        def cleanup():
            owner.close()
            while owner._leases:owner.release()
        self.addCleanup(cleanup)
        return owner

    def test_slow_copy_does_not_block_new_leases_or_unrelated_buffers(self):
        owner=self.owner()
        source=np.ones((3,4),np.uint16)
        entered,gate=threading.Event(),threading.Event()
        original=np.copyto
        def copy(destination,value):
            if value is source:
                entered.set()
                if not gate.wait(5):raise TimeoutError('copy gate')
            return original(destination,value)
        with ThreadPoolExecutor(3) as pool,patch('numpy.copyto',new=copy):
            slow=pool.submit(owner.prepare_named,dict(depth=source))
            try:
                self.assertTrue(entered.wait(2))
                pool.submit(owner.acquire).result(2)
                owner.release()
                other=pool.submit(owner.prepare_named,dict(gray=np.ones((2,3),np.uint8)))
                self.assertIn('gray',other.result(2))
                self.assertFalse(slow.done())
            finally:gate.set()
            self.assertIn('depth',slow.result(2))

    def test_same_buffer_is_published_once_when_two_preparations_race(self):
        owner=self.owner();source=np.ones((3,4),np.uint16)
        entered,gate,second_entered=threading.Event(),threading.Event(),threading.Event()
        original=np.copyto;copies=[]
        def copy(destination,value):
            copies.append(1);entered.set()
            if not gate.wait(5):raise TimeoutError('copy gate')
            return original(destination,value)
        def second():
            second_entered.set()
            return owner.prepare_named(dict(depth=source))
        with ThreadPoolExecutor(2) as pool,patch('numpy.copyto',new=copy):
            first=pool.submit(owner.prepare_named,dict(depth=source))
            try:
                self.assertTrue(entered.wait(2))
                another=pool.submit(second)
                self.assertTrue(second_entered.wait(2));self.assertFalse(another.done())
            finally:gate.set()
            self.assertEqual(first.result(2),another.result(2))
        self.assertEqual(1,len(copies));self.assertEqual(1,len(owner._blocks))

    def test_close_during_copy_discards_unpublished_handle_without_leak(self):
        owner=self.owner();source=np.ones((3,4),np.uint16)
        entered,gate=threading.Event(),threading.Event();names=[]
        original=np.copyto
        def copy(destination,value):
            entered.set()
            if not gate.wait(5):raise TimeoutError('copy gate')
            return original(destination,value)
        def create(*args,**kwargs):
            handle=SharedMemory(*args,**kwargs);names.append(handle.name);return handle
        with ThreadPoolExecutor(2) as pool,patch('numpy.copyto',new=copy), \
                patch('algorithm.registration_process.SharedMemory',side_effect=create):
            pending=pool.submit(owner.prepare_named,dict(depth=source))
            try:
                self.assertTrue(entered.wait(2))
                pool.submit(owner.close).result(2)
                with self.assertRaises(RuntimeError):owner.acquire()
            finally:gate.set()
            with self.assertRaisesRegex(RuntimeError,'released'):pending.result(2)
        self.assertEqual({},owner._blocks)
        for name in names:
            with self.assertRaises(FileNotFoundError):SharedMemory(name=name)


if __name__=='__main__':unittest.main()
