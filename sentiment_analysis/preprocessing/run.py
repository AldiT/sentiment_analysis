
import click

@click.command()
@click.option('--num-workers', default=16, help="The number of workers that will be used for preprocessing.")
def run(num_workers: int):
    print("Running the run function.")



if __name__ == "__main__":
    run()