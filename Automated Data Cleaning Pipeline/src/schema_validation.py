import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd

def get_schema():
    # Example schema: adjust according to your dataset
    schema = DataFrameSchema({
        "PassengerId": Column(int, Check(lambda x: x > 0), nullable=False),
        "Survived": Column(int, Check(lambda x: x.isin([0,1]))),
        "Pclass": Column(int, Check(lambda x: x.isin([1,2,3]))),
        "Name": Column(str, nullable=False),
        "Sex": Column(str, Check(lambda s: s.isin(["male", "female"]))),
        "Age": Column(float, Check(lambda x: x >= 0), nullable=True),  # Age can be missing or null
        "Fare": Column(float, Check(lambda x: x >= 0), nullable=True),
    }, coerce=True)
    return schema

def validate_schema(df: pd.DataFrame):
    schema = get_schema()
    # Validate and return errors if any
    return schema.validate(df, lazy=True)  # lazy=True aggregates all errors
