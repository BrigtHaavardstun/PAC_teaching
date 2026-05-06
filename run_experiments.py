# PAC_teaching/run_experiment.py
import sys
from OPENAI.err_extraction_low_deduc import run as run_err
from OPENAI.concept_identification_low_ind import run as run_concept
from OPENAI.err_analysis import run as run_analysis_err
from OPENAI.concept_analysis import run as run_analysis_concept
from OPENAI.combined_analysis import run as run_combined_analysis
from OPENAI.new_analysis import run as run_new_analysis
from OPENAI.new_analysis_2td import run as run_2td
if __name__ == "__main__":
    match sys.argv[1]:
        case "err":     run_err()
        case "concept": run_concept()
        case "err_analysis": run_analysis_err()
        case "concept_analysis": run_analysis_concept()
        case "combined_analysis": run_combined_analysis()
        case "new_analysis": run_new_analysis()
        case "2td": run_2td()
