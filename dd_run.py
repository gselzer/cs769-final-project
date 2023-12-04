import argparse
from techniques.datadiversification import data_diversification

parser = argparse.ArgumentParser(description='Data Diversification Training Script')
parser.add_argument('--k', type=int, required=True, help='k value')
parser.add_argument('--N', type=int, required=True, help='N value')
parser.add_argument('--n_epoch', type=int, required=True, help='Number of epochs')
parser.add_argument('--arch_fwd', type=str, required=True, help='Architecture forward')
parser.add_argument('--arch_bkwd', type=str, required=True, help='Architecture backward')
parser.add_argument('--src_lang', type=str, required=True, help='Source language')
parser.add_argument('--trg_lang', type=str, required=True, help='Target language')
parser.add_argument('--use_cpu', action='store_true', help='Use CPU only for training')

args = parser.parse_args()

dd = data_diversification.DataDiversification(k=args.k, N=args.N, n_epoch=args.n_epoch, use_cpu=args.use_cpu)
dd.diversify(
    arch_fwd=args.arch_fwd,
    arch_bkwd=args.arch_bkwd,
    src_lang=args.src_lang,
    trg_lang=args.trg_lang)

