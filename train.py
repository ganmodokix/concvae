import argparse
import os
import numpy as np

import yaml
import vaetc

import models
import evaluation

def evaluate_concvae(checkpoint: vaetc.Checkpoint, quantitative: bool = False):

    vaetc.evaluate(checkpoint, checkpoint, logging=True, qualitative=True, quantitative=quantitative)

    eval_data = vaetc.evaluation.preprocess.EncodedData(
        model=checkpoint.model,
        dataset=checkpoint.dataset.validation_set)
    np.savez(os.path.join(checkpoint.options["logger_path"], "zt_valid"), z=eval_data.z, t=eval_data.t)
    del eval_data
    test_data = vaetc.evaluation.preprocess.EncodedData(
        model=checkpoint.model,
        dataset=checkpoint.dataset.validation_set)
    np.savez(os.path.join(checkpoint.options["logger_path"], "zt_test"), z=test_data.z, t=test_data.t)
    del test_data
    
    evaluation.explain.visualize(checkpoint)
    evaluation.makesense.decisiontree.visualize(checkpoint)
    evaluation.makesense.randomforest.visualize(checkpoint)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--from_scratch", default=None, help="setting.yaml")
    parser.add_argument("--change_logger_path", default=None, help="overwrite `logger_path` option when in --from_scratch")
    parser.add_argument("--continue_exist", nargs=2, default=None, help="continue learning (checkpoint_path, extended_epochs) if specified")
    parser.add_argument("--evaluate", default=None, help="evaluate .pth if specified")

    args = parser.parse_args()
    
    if int(args.from_scratch is not None) + int(args.continue_exist is not None) + int(args.evaluate is not None) != 1:
        raise ValueError("Invalid options specified")

    if args.evaluate is not None:

        checkpoint = vaetc.load_checkpoint(args.evaluate)

    elif args.from_scratch is not None:
        
        with open(args.from_scratch, "r") as fp:
            options = yaml.safe_load(fp)
            options["hyperparameters"] = yaml.safe_dump(options["hyperparameters"])

        if args.change_logger_path is not None:
            options["logger_path"] = str(args.change_logger_path)
        
        checkpoint = vaetc.Checkpoint(options)

        vaetc.fit(checkpoint)

    elif args.continue_exist is not None:

        checkpoint_path, extended_epochs = args.continue_exist
        checkpoint = vaetc.load_checkpoint(checkpoint_path)
        
        vaetc.proceed(checkpoint, int(extended_epochs))

    evaluate_concvae(checkpoint, quantitative=True)