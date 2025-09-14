import argparse
from pathlib import Path
from utils.Io_utils import read_pairs_from_csv
from utils.output import output_Multiple, output_single
from utils.data_process import basepair_2_id_encode
from tests.inference import run_inference
from utils.evaluation import compute_AUPRC_and_AUROC_scores
from analysis.analysis import analysis_update

def parse_args():
     # 定义参数解析器
    parser = argparse.ArgumentParser(description="Demo: choose input from file or inline text")

    parser.add_argument("--module", choices=["test", "analysis"], required=True,
                        help="Select the module to run: 'test' for testing functionality, 'analysis' for data analysis")

    parser.add_argument("--source", choices=["file", "single"], required=False,
                        help="Specify input source: 'file' to read from a file, 'single' to use inline text input")
    parser.add_argument("--input_file", type=str, help="Path to input file (only valid when --source is 'file')")
    parser.add_argument("--sg", type=str, help="Inline sgRNA text content (only valid when --source is 'single')")
    parser.add_argument("--off", type=str, help="Inline off-target text content (only valid when --source is 'single')")

    parser.add_argument("--ori_path", type=str, help="Path to original matrix file (only valid when --module is 'analysis')")
    parser.add_argument("--update_path", type=str, help="Path to updated matrix file (only valid when --module is 'analysis')")
    parser.add_argument("--output_path", type=str, help="Path for output results (only valid when --module is 'analysis')")
    parser.add_argument("--seed_num", type=int, help="Number of random seeds (only valid when --module is 'analysis')")
    return parser.parse_args()


def load_input(args):
    if args.source == "file":
        if not args.input_file:
            raise ValueError("You chose --source file, but didn't provide --input_file.")
        file_path = Path(args.input_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        encode_list, label_list, sg_list = read_pairs_from_csv(file_path)
        predict_list, label_list, sg_list, test_sg_dict = run_inference(encode_list, label_list, sg_list)

        result_dict = compute_AUPRC_and_AUROC_scores(label_list, predict_list, sg_list, test_sg_dict)
        output_Multiple(result_dict)
  
    # source == "text"
    elif args.source == "single":
        sg = args.sg
        off = args.off 
        if not sg or not off:
            raise ValueError("Both --sg and --off must be provided when --source text.") 
            
        for seq, name in [(sg, "sg"), (off, "off")]:
            if len(seq) not in (23, 24):
                raise ValueError(f"--{name} length={len(seq)} is invalid, must be 23 or 24.")
        encode_list =[basepair_2_id_encode(sg, off)]
        label_list = [-1]
        sg_list = [sg[:-3]]
        predict_list, label_list, sg_list, test_sg_dict = run_inference(encode_list, label_list, sg_list)  
        output_single(sg, off, predict_list)

  
    

def main():
    args = parse_args()
    if args.module == "test":
        content = load_input(args)
    elif args.module == "analysis":
        analysis_update(args.ori_path, args.update_path, args.output_path, args.seed_num)



if __name__ == "__main__":
    main()