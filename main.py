import dapi as dp
import pandas as pd

def main():
    df = pd.read_csv('sample_anon.csv', index_col=0)
    df['time'] = pd.to_datetime(df['time'])
    dp.initialize(df, alias='sample_anon',
        min_responses=30, outlier_threshold=1.5, restrict_same_municipality=True,
        W=2, D=7)
    dp.display_state()
    dp.execute()


if __name__=="__main__":
    main()