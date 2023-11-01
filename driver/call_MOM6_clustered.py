from smartsim import Experiment
import glob
import itertools
from helpers import parse_mom6_out
import logging
import time
import pandas as pd
import shutil

logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
)

MOM6_NODES = 1
DB_NODES = 1

def main():
    timing_dfs = []

    mom6_cpus = [24,48,72,96,128]
    refines = [9] #Refinement relative to 1/4-degree, e.g. '3' refers to a 1/12-degree model

    combinations = itertools.product(mom6_cpus, refines)

    for mom6_cpu, refine in combinations:
        logging.info(f'Starting: \tmom6_cpu={mom6_cpu}')
        start_time = time.time()

        exp = Experiment(f"MOM6_run_exp_{mom6_cpu}_{refine}", launcher='slurm')
        # create and start an instance of the Orchestrator database
        # create and start an MOM6 experiment
        srun = exp.create_run_settings(
                exe="/scratch/gpfs/aeshao/dev/MOM6-examples/build/ocean_only/MOM6",
                run_command='srun'
                )
        srun.set_nodes(MOM6_NODES)
        srun.set_tasks(mom6_cpu)
        srun.set('cpu_bind','cores')
        srun.set_het_group([1])
        # start MOM6
        model = exp.create_model("MOM6_run", srun)
        model.params = {
            'nx':44*refine,
            'ny':40*refine,
            }

        files = glob.glob('/scratch/gpfs/aeshao/dev/MOM6-examples/ocean_only/double_gyre/*')
        model.attach_generator_files(to_copy=files, to_configure='./MOM_override')

        db = exp.create_database(
            db_nodes = DB_NODES,
            interface = ['ib0'],
        )

        db.set_cpus(12)
        db.set_run_arg('C', 'gpu')
        exp.generate(db, overwrite=True)

        exp.generate(model, overwrite=True)
        exp.start(db, model, summary=True, block=True)

        end_time = time.time()
        logging.info(f'Finished: \tmom6_cpu={mom6_cpu}\t{end_time-start_time}')

        exp.stop(db)
        tmp_df = parse_mom6_out()
        tmp_df['mom6_cpu'] = mom6_cpu
        tmp_df['refine'] = refine

        timing_dfs.append(tmp_df)

        time.sleep(2)

    full_df = pd.concat(timing_dfs, ignore_index=True)
    full_df.to_csv('timings.csv')

if __name__ == "__main__":
    main()
