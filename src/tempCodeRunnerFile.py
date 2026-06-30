missing = (

    df.isnull()
    .sum()

)

missing = (

    missing[missing > 0]

    .sort_values(
        ascending=False
    )

)

print(missing)