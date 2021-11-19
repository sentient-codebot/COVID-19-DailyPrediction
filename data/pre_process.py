import pandas as pd

perday_file = "https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv"
perday = pd.read_csv(perday_file, sep=";")
print(perday.head())

# select relevant
relevant_labels = ["Date_of_publication",
                    "Total_reported"]
selected = perday[relevant_labels]
national_perday = selected.groupby("Date_of_publication").sum()

print(national_perday)

# output .csv
output_file = "COVID-19_aantallen_nationale_per_dag.csv"
national_perday.to_csv("COVID-19_aantallen_nationale_per_dag.csv")