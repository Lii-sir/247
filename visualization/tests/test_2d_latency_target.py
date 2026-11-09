"""Latency goal reports use real request RTTs, never change quality results."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from detect_and_view import summarize_2d_latency


class TwoDLatencyTargetTests(unittest.TestCase):
    def rows(self,values):
        return [dict(cmd='detect_2d_piece',piece_id=i+1,duration_ms=value,ok=True)
                for i,value in enumerate(values)]

    def test_screenshot_values_identify_the_over_target_piece(self):
        rows=self.rows([774,845,833,922,942,829,827,789,890,1079])
        rows.append(dict(cmd='detect_3d_board',duration_ms=1503,ok=True))
        result=summarize_2d_latency(rows,expected_count=10)
        self.assertFalse(result['meets_target'])
        self.assertEqual([10],result['over_target_piece_ids'])
        self.assertEqual(1079,result['max_ms'])
        self.assertEqual(10,result['request_count'])

    def test_boundary_is_inclusive_and_failed_requests_cannot_pass_target(self):
        rows=self.rows([999.9,1000])
        self.assertTrue(summarize_2d_latency(rows,expected_count=2)['meets_target'])
        rows[0]['ok']=False
        self.assertFalse(summarize_2d_latency(rows,expected_count=2)['meets_target'])

    def test_incomplete_or_unavailable_measurements_are_not_reported_as_passed(self):
        self.assertIsNone(summarize_2d_latency([],expected_count=10)['meets_target'])
        self.assertFalse(summarize_2d_latency(self.rows([100]),expected_count=10)['meets_target'])
        invalid=self.rows([float('nan'),float('inf'),-1,True])
        self.assertIsNone(summarize_2d_latency(invalid,expected_count=4)['meets_target'])
        unknown=[dict(cmd='detect_2d_piece',piece_id=1,duration_ms=100)]
        self.assertFalse(summarize_2d_latency(unknown,expected_count=1)['meets_target'])


if __name__=='__main__':unittest.main()
