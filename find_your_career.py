import pandas as pd
import random

cols = {
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
}

skill_lvl = {
    4: "Average",
    5: "Intermediate",
    3: "Beginner",
    1: "Not Interested",
    6: "Excellent",
    2: "Poor",
    7: "Professional",
}


prev_df = pd.DataFrame()


def get_job_df(dataframe: pd.DataFrame, skill: str, user_skill: str | int):
    dataframe = dataframe[dataframe[skill] == user_skill]
    print(dataframe["Role"])
    return dataframe


if __name__ == "__main__":
    df = pd.read_csv("dataset9000.csv")
    flag = False
    for column in cols:
        select_something = int(
            input(f"Are you interested in {column}\n\
                1. Not Interested\n2. Poor\n3. Beginner\n4. Average\n5. Intermediate\n6. Excellent\n7. Professional\n")
        )
        selected = skill_lvl[select_something]
        prev_df = df.copy()

        print(len(set(df["Role"])))

        if len(set(df["Role"])) <= 10:
            break

        df = get_job_df(df, column, selected)

        if len(set(df["Role"])) == 0:
            break

    print(set(df["Role"]))
    print(len(set(df["Role"])))

    print(set(prev_df["Role"]))
    print(len(set(prev_df["Role"])))

    if len(set(prev_df["Role"])) > 5:
        listOfPrev_df = list(set(prev_df["Role"]))
        job_choices = []
        for i in range(3):
            choice = random.choice(listOfPrev_df)
            job_choices.append(choice)
            listOfPrev_df.remove(choice)
        print(job_choices)
