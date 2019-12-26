def imputer(dt,stgy):
    MisValLst = []
    for var in dt.columns:
        if dt[var].isnull().sum() > 0:
            Imputer = Imputer(missing_values=np.nan, strategy = stgy)
            dt[var] = Imputer.fit_transform(dt[var])

