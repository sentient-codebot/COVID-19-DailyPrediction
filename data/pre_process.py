import pandas as pd

perday_file = "COVID-19_aantallen_gemeente_per_dag.csv"
perday = pd.read_csv(perday_file, error_bad_lines=False)

grouped_muni = perday.groupby("Municipality_name")
# muni_perday = grouped_muni.sum()