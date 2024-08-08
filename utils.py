class Prepare():
    def __init__(self):
        pass

    def shuffle(self, df):
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        return shuffled_df