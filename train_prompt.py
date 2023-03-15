import argparse
from trainers import clip_prompts, eurosat, fgvcaircraft, dtd, caltech101, oxfordpets

def main(args):
    model, preprocess = clip_prompts()
    if args.dataset == 'eurosat':
        eurosat(model, preprocess, args.root)
    elif args.dataset == 'fgvcaircraft':
        fgvcaircraft(model, preprocess, args.root)
    elif args.dataset == 'dtd':
        dtd(model, preprocess, args.root)
    elif args.dataset == 'caltech101':
        caltech101(model, preprocess, args.root)
    elif args.dataset == 'oxfordpets':
        oxfordpets(model, preprocess, args.root)
    else:
        print('Not a valid dataset.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="", help="chosen dataset")

    args = parser.parse_args()
    main(args)
