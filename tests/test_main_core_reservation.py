"""Hybrid-core allocation checks with a simulated 8P+16E topology; no binding."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests')]
from algorithm.cpu_partition import choose_partition
from algorithm.config import load_runtime_config
from transport.bridge_server import AlgorithmServer
from pipeline_fakes import FakeDetector


class MainCoreReservationTests(unittest.TestCase):
    def cores(self):
        return ([dict(logical_cpus=[2*i,2*i+1],efficiency_class=2) for i in range(8)]
                +[dict(logical_cpus=[16+i],efficiency_class=0) for i in range(16)])

    def test_reservation_protects_entire_smt_core_and_can_use_other_cores(self):
        cores=[dict(logical_cpus=[0,1],efficiency_class=2),
               dict(logical_cpus=[2,3],efficiency_class=2),
               dict(logical_cpus=[4],efficiency_class=0)]
        partition=choose_partition(cores,range(5),2,reserved_cpus=[0])
        self.assertEqual([0,1],partition['main_cpus'])
        self.assertEqual([2,3,4],partition['registration_cpus'])
        with self.assertRaises(ValueError):choose_partition(cores,range(5),3,reserved_cpus=[0])
        with self.assertRaises(ValueError):choose_partition(cores,range(5),1,reserved_cpus=[99])

    def test_previous_ten_ten_layout_leaves_only_two_e_cores_for_main(self):
        cores=self.cores();allowed=list(range(32))
        for count in (10,2,10):
            plan=choose_partition(cores,allowed,count)
            allowed=plan['main_cpus']
        self.assertEqual([30,31],allowed)
        self.assertFalse(set(allowed).intersection(range(16)))

    def test_14900kf_layout_keeps_four_p_cores_for_main_and_two_for_two_d(self):
        cores=self.cores();available=list(range(32));starts=[];stops=[]
        server=AlgorithmServer();server.board=FakeDetector()
        self.addCleanup(server.close)
        def factory(role):
            def create(count,affinity,**kwargs):
                original=list(available)
                plan=choose_partition(cores,available,affinity['physical_cores'],
                                      reserved_cpus=affinity.get('reserved_main_cpus',()))
                available[:]=plan['main_cpus'];starts.append(role)
                service=Mock();service.info=dict(backend='process',workers=[dict(cpu_ids=plan['registration_cpus'])])
                def close():stops.append(role);available[:]=original
                service.close.side_effect=close
                return service
            return create
        path=ROOT/'config/config.json'
        config=load_runtime_config(path.read_text(encoding='utf-8'),str(path))
        settings=server._thread_settings(config)
        with patch('algorithm.registration_process.RegistrationProcesses',side_effect=factory('registration')), \
                patch('algorithm.cpu_2d_process.Cpu2DProcesses',side_effect=factory('cpu_2d')), \
                patch('algorithm.joint_process.JointMeasurementProcesses',side_effect=factory('joint_measurement')), \
                patch('algorithm.cpu_partition.physical_cores',return_value=cores), \
                patch('algorithm.cpu_partition.current_cpus',side_effect=lambda:list(available)):
            server._configure_cpu_processes(settings)
        info=server.threading_info
        groups={role:set(info[role+'_execution']['workers'][0]['cpu_ids'])
                for role in ('registration','cpu_2d','joint_measurement')}
        groups['main']=set(info['main_cpu_ids'])
        self.assertEqual(set(range(8)),groups['main'])
        self.assertEqual(set(range(8,12)),groups['cpu_2d'])
        self.assertEqual(set(range(12,18)),groups['registration'])
        self.assertEqual(set(range(18,32)),groups['joint_measurement'])
        self.assertEqual(set(range(8)),set(info['main_reserved_cpu_ids']))
        self.assertEqual('cpu_2d',starts[0])
        for role,cpus in groups.items():
            self.assertTrue(all(cpus.isdisjoint(other) for name,other in groups.items() if name!=role))
        server._stop_cpu_processes()
        self.assertEqual(list(reversed(starts)),stops)
        self.assertEqual(list(range(32)),available)

    def test_invalid_reservation_values_are_rejected_without_native_calls(self):
        server=AlgorithmServer.__new__(AlgorithmServer)
        for value in (-1,True,1.5,'4',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                server._thread_settings(dict(common_3d=dict(parameters=dict(threading=dict(
                    main_reserved_physical_cores=value)))))
        self.assertEqual(0,server._thread_settings({})['main_reserved_physical_cores'])

    def test_failed_3d_startup_restores_two_d_allocation(self):
        cores=self.cores();available=list(range(32));server=AlgorithmServer();server.board=FakeDetector()
        self.addCleanup(server.close)
        service=Mock();service.info=dict(backend='process')
        def create(count,affinity,**kwargs):
            plan=choose_partition(cores,available,count,reserved_cpus=affinity['reserved_main_cpus'])
            available[:]=plan['main_cpus']
            service.close.side_effect=lambda:available.__setitem__(slice(None),list(range(32)))
            return service
        path=ROOT/'config/config.json'
        settings=server._thread_settings(load_runtime_config(path.read_text(encoding='utf-8'),str(path)))
        with patch('algorithm.cpu_partition.physical_cores',return_value=cores), \
                patch('algorithm.cpu_partition.current_cpus',side_effect=lambda:list(available)), \
                patch('algorithm.cpu_2d_process.Cpu2DProcesses',side_effect=create), \
                patch('algorithm.registration_process.RegistrationProcesses',side_effect=RuntimeError('3D startup failed')):
            with self.assertRaisesRegex(RuntimeError,'3D startup failed'):server._configure_cpu_processes(settings)
        service.close.assert_called_once()
        self.assertEqual(list(range(32)),available)


if __name__=='__main__':unittest.main()
