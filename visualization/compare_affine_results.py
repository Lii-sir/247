"""Compare two saved lab/formal reports; never run inference or registration."""
import argparse
import json
from pathlib import Path


def records(document):
    if 'stages' in document and 'inputs' in document:
        result=next((stage['registration'] for stage in reversed(document['stages'])
                     if 'registration' in stage),None)
        if result is not None:
            return {int(document['inputs'].get('piece_id',1)):(result,document.get('timings',{}).get('仿射配准'))}
    response=document.get('response',document)
    pieces=response.get('results_board',{}).get('pieces',[])
    output={}
    for piece in pieces:
        if 'registration' in piece:
            output[int(piece['piece_id'])]=(piece['registration'],piece.get('timings',{}).get('affine_registration_ms'))
    if not output:raise ValueError('No per-piece affine records found')
    return output


def compare(first,second):
    a,b=records(first),records(second)
    if a.keys()!=b.keys():raise ValueError('Baseline and candidate piece IDs must match')
    rows=[]
    fields=('affine_function_evaluations','affine_warp_evaluations','affine_cache_hit_rate',
            'affine_optimizer_ms','affine_score_after','affine_zone_losses','affine_valid',
            'affine_initialization','affine_multiscale','affine_budget')
    for pid in sorted(a):
        left,lms=a[pid];right,rms=b[pid]
        row=dict(piece_id=pid,baseline_ms=lms,candidate_ms=rms,
            baseline={key:left.get(key) for key in fields},candidate={key:right.get(key) for key in fields})
        if isinstance(lms,(int,float)) and isinstance(rms,(int,float)) and lms>0:
            row['wall_reduction_fraction']=1-rms/lms
        lm,rm=left.get('matrix_2d_to_3d'),right.get('matrix_2d_to_3d')
        row['matrix_identical']=lm==rm if lm is not None and rm is not None else None
        rows.append(row)
    return dict(note='Compare same inputs, calibration, tolerances and concurrency. Quality still needs overlay/height review.',pieces=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=compare(json.loads(args.baseline.read_text(encoding='utf-8-sig')),
                   json.loads(args.candidate.read_text(encoding='utf-8-sig')))
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding='utf-8')
    else:print(text)


if __name__=='__main__':main()
