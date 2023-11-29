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

def main():
    timing_dfs = []

    case_root = '/scratch/gpfs/aeshao/dev/Forpy_CNN_GZ21/driver/cases/OM4_025'

    db_nodes_list = [1]
    mom6_nodes_list = [1,3,5]

    combinations = itertools.product(mom6_nodes_list, db_nodes_list)

    for mom6_nodes, db_nodes in combinations:
        logging.info(f'Starting: \tdb_nodes={db_nodes}\tmom6_nodes={mom6_nodes}')
        start_time = time.time()

        exp = Experiment(f"OM4_025_{db_nodes}_{mom6_nodes}", launcher='slurm')
        # create and start an instance of the Orchestrator database
        # create and start an MOM6 experiment
        srun = exp.create_run_settings(
                exe="/scratch/gpfs/aeshao/dev/MOM6-examples/build/ice_ocean_SIS2/MOM6",
                run_command='srun'
                )

        ncpus = mom6_nodes*128
        srun.set_nodes(mom6_nodes)
        srun.set_tasks(ncpus)
        srun.set('cpu_bind','cores')
        srun.set_het_group([1])
        # start MOM6
        model = exp.create_model("MOM6_run", srun)

        clustered_value = 'False'
        if db_nodes > 1:
            clustered_value = 'True'
        model.params = {
                'clustered':clustered_value
            }

        files = glob.glob(f'{case_root}/base/')
        files.append(f'{case_root}/base/RESTART')
        sym_files = [f'{case_root}/input/INPUT']
        shutil.copy("MOM_override.OM4_025", "MOM_override")
        model.attach_generator_files(to_copy=files, to_configure='./MOM_override', to_symlink=sym_files)
        model.add_script('pys', script_path='/scratch/gpfs/aeshao/dev/Forpy_CNN_GZ21/testNN_trace.txt')
        model.add_ml_model(
                f'ml_0',
                'TORCH',
                model_path='/scratch/gpfs/aeshao/dev/Forpy_CNN_GZ21/CNN_GPU_2X21X21X2.pt',
                device=f'GPU',
                batch_size=ncpus,
                min_batch_size=ncpus
        )
        db = exp.create_database(
            db_nodes = db_nodes,
            interface = ['ib0'],
        )

        db.set_cpus(48)
        db.set_run_arg('C', 'gpu')
        exp.generate(db, overwrite=True)

        exp.generate(model, overwrite=True)
        exp.start(db, model, summary=True, block=True)

        end_time = time.time()
        logging.info(f'Finished: \tdb_nodes={db_nodes}\t{end_time-start_time}')

        exp.stop(db)
#        tmp_df = parse_mom6_out()
#        tmp_df['db_nodes'] = db_nodes
#        tmp_df['ncpus'] = ncpus
#
#        timing_dfs.append(tmp_df)

        time.sleep(2)

    full_df = pd.concat(timing_dfs, ignore_index=True)
    full_df.to_csv('timings.csv')

if __name__ == "__main__":
    main()
