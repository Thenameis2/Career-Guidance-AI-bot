import pandas as pd


cols = [
    "Database Fundamentals",
    "Computer Architecture",
    "Distributed Computing Systems",
    "Cyber Security",
    "Networking",
    "Software Development",
    "Programming Skills",
    "Project Management",
    "Computer Forensics Fundamentals",
    "Technical Communication",
    "AI ML",
    "Software Engineering",
    "Business Analysis",
    "Communication skills",
    "Data Science",
    "Troubleshooting skills",
    "Graphics Designing",
]


def get_job_df(dataframe: pd.DataFrame, skill: str, user_skill):
    flag = False
    if len(dataframe) <= 10:
        flag = True
        return dataframe, flag
    dataframe = dataframe.loc[dataframe[skill] == user_skill]
    return dataframe, flag
