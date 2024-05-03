# LTV Model Creation Script:
from datetime import datetime,timedelta

def create_model(client_id, user_id, process_id, upload_id, avg_lf, model_type, process_mode, skin_code):
    # ==================================================================
    # Importing Packages
    # ==================================================================
    import os
    import matplotlib
    matplotlib.use('Agg')
    import json
    import pickle
    import sys
    import warnings
    import numpy as np
    import pandas as pd
    import vertica_python
    import feature_selector
    import plotly.express as px
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    warnings.filterwarnings('ignore')
    import xgboost
    import lightgbm as lgb
    import analytics_base_functions as abf
    import analytics_plot_functions as apf
    import analytics_model_functions as amf
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn import ensemble, metrics, linear_model, neighbors, tree
    start_time = abf.current_date_time()
    import plotly.io as pio
    scope = pio.kaleido.scope
    # ==================================================================
    # Connecting with DB
    # ==================================================================
    # Reading envronment variable and config files
    acc_home = os.environ['ACCELERATOR_HOME']
    acc_home = acc_home+'/'
    with open(acc_home+'python_scripts/config.json', 'r') as json_file:
        configs = json.load(json_file)
    with open(acc_home+'python_scripts/notification_query.json', 'r') as json_file:
        noti_query_config = json.load(json_file)
    
    #log file initiation
    logger = abf.log_file_initiate(acc_home,configs,client_id,user_id,process_id)
    logger.info("Logger initialized successfully.")

    with open(acc_home+'python_scripts/analytics_stage_query.json', 'r') as json_file:
        analytics_query_config = json.load(json_file)
    with open(acc_home+'sql_scripts/vertica_schema_set.sql', 'r') as sql_file:
        schema_set_query = sql_file.read()
    schema_qualifier = os.environ['SCHEMA_QUALIFIER']

    # using the env variable to switch to dev schemas here
    schema_type = configs['schema_type'][schema_qualifier]
    analytics_schema = schema_type['analytics']
    # Connection details for Vertica DB:
    conn_info = abf.vertica_connection(configs)
    # =>Connecting to Vertica DB
    vertica_connection = vertica_python.connect(**conn_info)
    vsql_cur = vertica_connection.cursor()
    logger.info("Successfully connected to Vertica DB")
    # Statement for extracting data
    process_id_var = str(process_id)
    client_id_var = str(client_id)
    # Fetching data from Vertica DB
    vsql_cur.execute(schema_set_query, {'schema_name': analytics_schema})
    vsql_cur.execute(analytics_query_config['DATA_QUERY']['LTV'], {'process_id_var': process_id_var, 'client_id_var': client_id_var})
    # Converting into DataFrame formate
    data = pd.DataFrame(vsql_cur.fetchall())
    logger.info("Fetching data from DB is Completed")
    data.columns = [c.name for c in vsql_cur.description]
    logger.info(list(data.columns))
    vertica_connection.close()

    # Connection details for mysql DB
    mysql_connection = abf.mysql_connection(configs)

    logger.info("Successfully connected to MySQL DB")
    # Reading out the model_id from process table
    model_id = abf.mysql_fetch(mysql_connection,analytics_query_config['MODEL_ID_FETCH'], (process_id_var,))
    # Reading test size data from process table
    test_data_size = abf.mysql_fetch(mysql_connection,analytics_query_config['TEST_DATA_SIZE_FETCH'], (int(model_id),))
    target_var = abf.mysql_fetch(mysql_connection, analytics_query_config['TARGET_VAR_FETCH'], (int(model_id),))
    rfm_seg_col = target_var.capitalize() + '_Overall'

    if test_data_size in [20, 25, 30, 40]:
        test_data_size = test_data_size/100
    else:
        test_data_size = 0.2

    # No Samples in training model
    if int(data.shape[0]) < 1:
        logger.info("No Data for Model Creation")
        abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_ERROR_IN_MODEL_TABLE'],("Error", abf.current_date_time(), 0, model_id))
        sys.exit()
  # ==================================================================
  # Feature Engineering
  # ==================================================================
    # Dropping duplicate records
    logger.info("Data Preparation is Started")
    Data = data
    Data.drop_duplicates(keep='first', inplace=True)
    TP = Data['prediction_period'].unique()[0]
    logger.info(TP)
    Data.drop(['prediction_period'], axis=1)

    # Checking Outlier data.
    Data.rename(columns={"actual_ngr":target_var},inplace=True)
    Data[target_var] = np.where(Data[target_var].isnull(), 0, Data[target_var])
    Distinct_Player_Count = int(Data.Customer_Id.nunique())
    Q1 = Data[target_var].quantile(0.025)
    Q3 = Data[target_var].quantile(0.975)
    Data_out = Data[((Data[target_var] < Q1) | (Data[target_var] > Q3))]
    Data = Data[~((Data[target_var] < Q1) | (Data[target_var] > Q3))]

    # Updating Process Table
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS_FIRST'],(0, 6, "Analytics", abf.current_date_time(), process_id))
    # Start Feature Engineering/Selection(process_log table)
    abf.mysql_execute(mysql_connection,analytics_query_config['INSERT_PROCESS_LOG'], (abf.current_date_time(), "Analytics Layer", 0, "Feature Engineering", abf.current_date_time(), process_id))
    # More Missing values in Target Variable
    if (Data[target_var].isnull().sum() > (0.25 * Data.shape[0])):
        logger.info("More number of missing values in target variables")
        abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_ERROR_IN_MODEL_TABLE'],("Error", abf.current_date_time(), 0, model_id))
        sys.exit()

    # #### Data Preparation :
    lis_var = []
    # Dropping More than 25% Missing value variables.
    for i in Data.columns.difference(['Customer_Id', 'prediction_period', rfm_seg_col, 'Days_Since_Last_Bet', 'Apd_Overall']):
        if Data[str(i)].isnull().sum() > (0.25 * Data.shape[0]):
            lis_var.append(str(i))
            Data.drop(str(i), axis=1, inplace=True)

    # Defining Path
    root = acc_home+configs['home']['result']
    dirname = "{}/{}/{}".format(client_id, user_id, process_id)
    dirpath = os.path.join(root, dirname)
    # Saving all dropped variables
    Data_out.to_csv(dirpath+"/Outlier_Data.csv", index=False)
    Droped_Var = pd.DataFrame()
    Droped_Var['Columns'] = lis_var
    Droped_Var.to_csv(dirpath+"/Droped_Var.csv", index=False)

    if int(Data.shape[1]) < 1:
        logger.info("All variables has missing values")
        abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_ERROR_IN_MODEL_TABLE'],("Error", abf.current_date_time(), 0, model_id))
        sys.exit()

    # Checking distribution of target variable
    if (len(Data[target_var].value_counts()) < 2):
        logger.info("Target variable has no variance")
        abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_ERROR_IN_MODEL_TABLE'],("Error", abf.current_date_time(), 0, model_id))
        sys.exit()
    # No features for model creation
    if int(Data.shape[1]) < 1:
        logger.info("All variables has missing values")
        abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_ERROR_IN_MODEL_TABLE'],("Error", abf.current_date_time(), 0, model_id))
        sys.exit()

    # Seperating Numerical and categorical data type column names
    num_col = [key for key in dict(Data.dtypes) if dict(Data.dtypes)[key] in ['int64', 'int32', 'float64', 'float32']]
    cat_col = [key for key in dict(Data.dtypes) if dict(Data.dtypes)[key] in ['object']]

    # Saving numerical and categorical features
    with open(dirpath+'/'+"Features_Num.bin", "wb") as output:
        pickle.dump(num_col, output)

    with open(dirpath+'/'+"Features_Cat.bin", "wb") as output:
        pickle.dump(cat_col, output)

    # Seperating Numerical and categorical variables
    Data_Num = Data[num_col]
    Data_Cat = Data[cat_col]

    # Creating an object for imputing missing values
    imputer = SimpleImputer(strategy="median")
    Num_MISS = imputer.fit(Data_Num)

    # Saving Objects:
    filename = "/Missing_Impution_Object.sav"
    pickle.dump(Num_MISS, open(dirpath+filename, 'wb'))
    Data_NUM = pd.DataFrame(Num_MISS.transform(Data_Num))
    Data_NUM.columns = Data_Num.columns
    logger.info("Imputing Missing Values is Completed")
    # #### Outlier Treatment :
    for col in Data_NUM.columns.difference(['Customer_Id', 'prediction_period']):
        percentiles = Data_NUM[col].quantile([0.01, 0.99]).values
        Data_NUM[col] = np.clip(Data_NUM[col], percentiles[0], percentiles[1])
    logger.info("Outlier treatment is Completed")

    # ====> Categorical Data :
    # Imputing Missing values
    for i in Data_Cat.columns.difference(['Customer_Id']):
        Data_Cat[str(i)].fillna(Data_Cat[str(i)].mode()[0], inplace=True)
    # Identifying More categories categorical variables
    dd = []
    for i in Data_Cat.columns.difference(['Customer_Id']):
        if(len(Data_Cat[str(i)].value_counts())) > 4:
            dd.append(i)

    # Reducing Categories
    Cat = pd.DataFrame()
    for i in dd:
        Val = Data_Cat[str(i)].value_counts().index
        Val = Val[0:3]
        Cat[str(i)] = Val[0:3]
        Data_Cat[str(i)] = np.where(Data_Cat[str(i)] == Val[0], Val[0],
                                    np.where(Data_Cat[str(i)] == Val[1], Val[1],
                                             np.where(Data_Cat[str(i)] == Val[2], Val[2], 'Others')))

    Cat.to_csv(dirpath+"/Categories_LTV.csv")

    # Creating dummy variable for categorical variables
    for c_feature in Data_Cat.columns.difference(['Customer_Id']):
        Data_Cat[c_feature] = Data_Cat[c_feature].astype('category')
        Data_Cat = abf.create_dummies(Data_Cat, c_feature)
    logger.info("Reduced categories and Dummy variable creation is completed")
    # Combining both numerical and categorical data
    Data_NUM.reset_index(drop=True, inplace=True)
    Data_Cat.reset_index(drop=True, inplace=True)
    DATA = pd.concat([Data_NUM, Data_Cat], axis=1)

    # Symbols are not considered in Column names. Replacing with 2
    DATA.columns = DATA.columns.str.strip()
    DATA.columns = DATA.columns.str.replace(' ', '_')
    DATA.columns = DATA.columns.str.replace(r"[^a-zA-Z\d\_]+", "")
    DATA.columns = DATA.columns.str.replace(r"[^a-zA-Z\d\_]+", "")
    logger.info("Data Preparation is Completed")
    DATA1 = DATA

    # Calculating Quantiles for variables
    quantiles = DATA1[['Days_Since_Last_Bet', 'Apd_Overall', rfm_seg_col]].quantile(q=[0.25, 0.5, 0.75])
    quantiles = quantiles.to_dict()
    filename = "/RFM_Quantile_LTV.sav"
    pickle.dump(quantiles, open(dirpath+filename, 'wb'))
    # User defined function for Creating segments using Recency Variable
    # Arguments (x = value, p = recency, frequency,monetary_value, k = quartiles dict)

    # Implementing user defined function for creating segments.
    DATA1['R_Quartile'] = DATA1['Days_Since_Last_Bet'].apply(amf.RClass, args=('Days_Since_Last_Bet', quantiles,))
    DATA1['F_Quartile'] = DATA1['Apd_Overall'].apply(amf.FClass, args=('Apd_Overall', quantiles,))
    DATA1['M_Quartile'] = DATA1[rfm_seg_col].apply(amf.MClass, args=(rfm_seg_col, quantiles,))
    # Overall Quartile RFM Segments
    DATA1['Overall_RFMQuartile_Score'] = DATA1['R_Quartile'] + DATA1['F_Quartile'] + DATA1['M_Quartile']
    # Creating aggregated by FRM Quantile score
    DATA_Q_Rank = DATA1.groupby('Overall_RFMQuartile_Score', as_index=False).agg({"Days_Since_Last_Bet": "mean", "Apd_Overall": "mean", rfm_seg_col : 'mean', "Customer_Id": "count"})
    # Calculating M Quantiles for variables
    M_quantiles = DATA_Q_Rank[[rfm_seg_col]].quantile(q=[0.25, 0.5, 0.75])
    M_quantiles = M_quantiles.to_dict()
    # Saving M_Qunatile
    filename = "/M_Quantile_LTV.sav"
    pickle.dump(M_quantiles, open(dirpath+filename, 'wb'))

    # Creating segment based on qunatile score
    DATA_Q_Rank['Segment'] = 0
    DATA_Q_Rank.loc[DATA_Q_Rank[rfm_seg_col] <= DATA_Q_Rank[rfm_seg_col].quantile(0.75), 'Segment'] = 1
    DATA_Q_Rank.loc[DATA_Q_Rank[rfm_seg_col] <= DATA_Q_Rank[rfm_seg_col].quantile(0.50), 'Segment'] = 2
    DATA_Q_Rank.loc[DATA_Q_Rank[rfm_seg_col] <= DATA_Q_Rank[rfm_seg_col].quantile(0.25), 'Segment'] = 3
    DATA1 = pd.merge(DATA1, DATA_Q_Rank[["Overall_RFMQuartile_Score", "Segment"]], how="left", on=["Overall_RFMQuartile_Score"])
    logger.info("Player Segmentation is Completed")
    # Train and Test data for feature selection
    logger.info("Feature Selection is Started")
    TRAIN, TEST = train_test_split(DATA1, test_size=test_data_size, random_state=1234)
    # Selecting Average Lifetime Value Period
    train_y = TRAIN[target_var]
    AVG_NGR_VALUE = np.round(TRAIN[target_var].mean(), 4)

  # =======================================================
  # #### Feature Selection :
  # =======================================================
    # Seperating Dependent and Independent variables
    FS = feature_selector.FeatureSelector(data=TRAIN[TRAIN.columns.difference(['Customer_Id', 'prediction_period', 'actual_apd',target_var, 'R_Quartile', 'F_Quartile', 'M_Quartile', 'Overall_RFMQuartile_Score', 'Segment'])], labels=TRAIN[str(train_y.name)])
    FS.identify_zero_importance(task='regression', eval_metric='l2', n_iterations=3, early_stopping=False)
    # Listing all significant features
    Val_Feature = FS.feature_importances
    Val_Feature = Val_Feature.sort_values(['importance'], ascending=False)
    Val_Feature['Check'] = np.where(Val_Feature.cumulative_importance < 0.75, 1, 0)
    Val_Feature = Val_Feature[['feature', 'normalized_importance', 'Check']]
    logger.info(Val_Feature)
    VAL = Val_Feature[Val_Feature.Check == 1]
    logger.info(VAL)
    logger.info(VAL.shape[0])
    if int(VAL.shape[0]) > 0:
        model_Ngr = list(VAL.feature)
    else:
        model_Ngr = list(Val_Feature['feature'].head(35))

    all_features_lis_LTV = list(Val_Feature['feature'])

    # Saving All feature variables
    with open(dirpath+'/'+"All_Features_LTV.bin", "wb") as output:
        pickle.dump(all_features_lis_LTV, output)
    with open(dirpath+'/'+"All_Features_LTV.bin", "rb") as data:
        All_Features_LTV = pickle.load(data)

    # Saving Important variables
    with open(dirpath+"/Features_Ngr.bin", "wb") as output:
        pickle.dump(model_Ngr, output)
    with open(dirpath+"/Features_Ngr.bin", "rb") as data:
        Features_Ngr = pickle.load(data)

    # Storing Top features and plotting thier significance values
    Val_Feature_10 = Val_Feature.head(10)
    Val_Feature_5 = Val_Feature.head()
    Val_Feature_5 = Val_Feature_5.sort_values(['normalized_importance'], ascending=True)
    x = list(Val_Feature_5.feature)
    
    #calling key feature plot function
    apf.key_feature_plot(x,list(Val_Feature_5.normalized_importance),dirpath)

    # Saving Top-5 features
    with open(dirpath+"/Features_Ngr_5.bin", "wb") as output:
        pickle.dump(x, output)

    # Update Feature Selection Completion(process_log table)
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS_LOG'], (1,abf.current_date_time(), "Feature Engineering", process_id))
    # Updating Process Table
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS'],(1, 'Running', abf.current_date_time(), process_id))
    logger.info("Feature Selection is Completed")

    # ==================================================================
    # Exploratory Data Analysis
    # ==================================================================
    K = dict()
    for i in list(Val_Feature_10.feature):
        logger.info(f"Started EDA for {i} variable")
        ED_Summ_Data_1 = dict()
        ED_Summ_Data_2 = dict()
        I = i.replace('_', ' ').title()
        No_Uniq= len(DATA1[str(i)].value_counts())
        if No_Uniq <= 8:
            #Calculating Basic Stats
            K[str(I)] = dict(Name=str(I),No_Observations=DATA1[str(i)].count(),N_Unique=len(DATA1[str(i)].value_counts()),Mode=DATA1[str(i)].mode()[0],Values=list(DATA1[str(i)].value_counts().index),Frequency=list(DATA1[str(i)].value_counts().values))
            
            #Key feature distribution plot
            labels = list(DATA1[str(i)].value_counts().index)
            values = list(DATA1[str(i)].value_counts().values)
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent')])
            fig.update_layout(title=f'{I} Distribution',title_x=0.5,autosize=False,width=450,height=450)
            # fig.update_traces(marker=dict(colors=['#5DC2CD','#2266BB','#49AFBB','#5889AF','#007FC5','#66C2A3','#3A9C92','#248989','#1284A7','#434DB0']))
            fig.update_traces(marker=dict(colors=["#7BAFBA", "#0C4F9F", "#5960A5", "#1F26CD", "#B4DFFF", "#7FD7DB", "#00CEFF", "#908DFF", "#169EC7", "#0278AD", "#144F8D", "#6053F8"]))
            config = {'displayModeBar': False}
            fig.write_html(f"{dirpath}/{I.replace(' ', '_')}_1.html",config=config)
            fig.write_image(f"{dirpath}/{I.replace(' ', '_')}_1.png")
            scope._shutdown_kaleido()
            ED_Summ_Data_1['Feature_Name'] = str(I)
            ED_Summ_Data_1['Graph_Type'] = 'Pie'
            ED_Summ_Data_1['Label_Names'] = labels
            ED_Summ_Data_1['Label_Values'] = values
            ED_Summ_Data_1 = json.dumps(ED_Summ_Data_1, default=abf.myconverter)
            json.dump(ED_Summ_Data_1, open(f'{dirpath}/{i}_1.json', 'w'))
        
            #Relation between key feature and target feature(Continuous and Categorical feature)
            fig = px.box(DATA1, x=target_var, y=str(i),height=450,width=450,title=f'Actual {target_var.upper()} vs {I}', color_discrete_sequence = ["#0C4F9F", '#0278AD', '#169EC7'])
            fig.update_layout(title_x=0.5)
            fig.update_yaxes(title_text=str(I))
            fig.update_xaxes(title_text=target_var.replace("_"," ").title())
            config = {'displayModeBar': False}
            fig.write_html(f"{dirpath}/{I.replace(' ', '_')}_2.html",config=config)
            fig.write_image(f"{dirpath}/{I.replace(' ', '_')}_2.png")
            scope._shutdown_kaleido()
            ED_Summ_Data_2['Feature_Name'] = str(I)
            ED_Summ_Data_2['Graph_Type'] = 'BoxPlot'
            ED_Summ_Data_2['Independent_Col'] = list(DATA1[str(i)])
            ED_Summ_Data_2['Dependent_Col'] = list(DATA1[target_var])
            ED_Summ_Data_2 = json.dumps(ED_Summ_Data_2, default=abf.myconverter)
            json.dump(ED_Summ_Data_2, open(f'{dirpath}/{i}_2.json', 'w'))
        
        else:
            #Calculating Basic Stats
            K[str(I)] = dict(Name=str(I),No_Observations=DATA1[str(i)].count(),N_Unique=len(np.unique(DATA1[str(i)])),Min=round(DATA1[str(i)].min(),2),Average=round(DATA1[str(i)].mean(),2),Median=round(DATA1[str(i)].median(),2),Max=round(DATA1[str(i)].max(),2))
            
            #Key feature distribution plot  
            plt.figure(figsize=(4,4))
            fig = px.histogram(DATA1, x=str(i),title=f'{I} Distribution',height=450,width=450,nbins=30, color_discrete_sequence = ["#0C4F9F", '#0278AD', '#169EC7'])
            fig.update_layout(title_x=0.5)
            fig.update_yaxes(title_text='Player Count')
            fig.update_xaxes(title_text=str(I))
            config = {'displayModeBar': False}
            fig.write_html(f"{dirpath}/{I.replace(' ', '_')}_1.html",config=config)
            fig.write_image(f"{dirpath}/{I.replace(' ', '_')}_1.png")
            scope._shutdown_kaleido()
            ED_Summ_Data_1['Feature_Name'] = str(I)
            ED_Summ_Data_1['Graph_Type'] = 'Histogram'
            ED_Summ_Data_1['Independent_Col'] = list(DATA1[str(i)])
            ED_Summ_Data_1 = json.dumps(ED_Summ_Data_1, default=abf.myconverter)
            json.dump(ED_Summ_Data_1, open(f'{dirpath}/{i}_1.json', 'w'))
        
            #Relation between key feature and target feature(Both variables are Continuous)
            fig = px.scatter(DATA1,x=str(i), y=target_var,height=450,width=450,title=f'Actual {target_var.upper()} vs {I}', color_discrete_sequence = ["#0C4F9F"])
            fig.update_layout(title_x=0.5)
            fig.update_xaxes(title_text=str(I))
            fig.update_yaxes(title_text=target_var.replace("_"," ").title())
            config = {'displayModeBar': False}
            fig.write_html(f"{dirpath}/{I.replace(' ', '_')}_2.html",config=config)
            fig.write_image(f"{dirpath}/{I.replace(' ', '_')}_2.png")
            scope._shutdown_kaleido()
            ED_Summ_Data_2['Feature_Name'] = str(I)
            ED_Summ_Data_2['Graph_Type'] = 'ScatterPlot'
            Plt_Summarise_data = DATA1.groupby([str(i)],as_index=False)[target_var].mean()
            ED_Summ_Data_2['Independent_Col'] = list(Plt_Summarise_data[str(i)])
            ED_Summ_Data_2['Dependent_Col'] = list(Plt_Summarise_data[target_var])
            ED_Summ_Data_2 = json.dumps(ED_Summ_Data_2, default=abf.myconverter)
            json.dump(ED_Summ_Data_2, open(f'{dirpath}/{i}_2.json', 'w'))

        del ED_Summ_Data_1, ED_Summ_Data_2
        logger.info(f"Started EDA for {i} variable")

    #Saving dictionary file and EDA data
    EDA_Data = DATA1[Val_Feature_10.feature]
    EDA_Data['actual_ltv'] = DATA1[target_var]
    EDA_Data.columns = EDA_Data.columns.str.replace("_", " ").str.title()
    EDA_Data.to_csv(dirpath+'/EDA_Data.csv',index=False)

    K = eval(str(K))
    with open(dirpath+"/Feature_EDA.json", "w") as outfile:
        json.dump(K, outfile)

  # ==================================================================
  # Train and Test Split :
  # ==================================================================
    # Update 'Train and Test Split started'(process_log table)
    logger.info("Train and Test Split is Started")
    abf.mysql_execute(mysql_connection,analytics_query_config['INSERT_PROCESS_LOG'], (abf.current_date_time(), "Analytics Layer", 0, "Train Test Split", abf.current_date_time(), process_id))

    # Calling save_input_data function from analytics_model_function module store model input metadata
    amf.save_input_data(TRAIN, All_Features_LTV, model_id, client_id, analytics_query_config, conn_info, schema_set_query, analytics_schema)

    # Seperating independent and dependent variables
    TRAIN_X = TRAIN[Features_Ngr]
    TRAIN_Y = TRAIN[train_y.name]
    TEST_X = TEST[Features_Ngr]
    TEST_Y = TEST[train_y.name]
    # Update Train and Test Split completed'(process_log table)
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS_LOG'], (1,abf.current_date_time(), "Train Test Split", process_id))
    # Updating Process Table
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS'],(2, 'Running', abf.current_date_time(), process_id))
    logger.info("Train and Test Split is Completed")
  # ==================================================================
  # Model Algorithm Selection :
  # ==================================================================
    logger.info("Model Algorithm Selection is Started")
    # Inserting 'Model Alogirthm Selection' started(process_log table)
    abf.mysql_execute(mysql_connection,analytics_query_config['INSERT_PROCESS_LOG'], (abf.current_date_time(), "Analytics Layer", 0, "Model Selection", abf.current_date_time(), process_id))

    # ### Model Building :
    MLA = [
        ensemble.AdaBoostRegressor(),
        ensemble.BaggingRegressor(),
        ensemble.GradientBoostingRegressor(),
        ensemble.RandomForestRegressor(),
        xgboost.XGBRegressor(objective='reg:squarederror'),
        linear_model.LinearRegression(),
        neighbors.KNeighborsRegressor(),
        tree.DecisionTreeRegressor(),
        lgb.LGBMRegressor()
    ]

    MLA_columns = []
    MLA_compare = pd.DataFrame(columns=MLA_columns)

    # Calculating metrics(Train and Test Accuracy, Train and Test Mean Squared Error) for all models
    row_index = 0
    for alg in MLA:
        alg = alg.fit(TRAIN_X, TRAIN_Y.values.ravel())
        predicted_train = alg.predict(TRAIN_X)
        predicted_test = alg.predict(TEST_X)
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA_Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA_Train_Accuracy'] = round(alg.score(TRAIN_X, TRAIN_Y), 4)
        MLA_compare.loc[row_index, 'MLA_Test_Accuracy'] = round(alg.score(TEST_X, TEST_Y), 4)
        MLA_compare.loc[row_index, 'MLA_Train_Mse'] = metrics.mean_squared_error(TRAIN_Y, predicted_train)
        MLA_compare.loc[row_index, 'MLA_Test_Mse'] = metrics.mean_squared_error(TEST_Y, predicted_test)
        row_index += 1

    MLA_compare = MLA_compare[MLA_compare.MLA_Train_Accuracy > 0]
    MLA_compare.sort_values(by=['MLA_Test_Accuracy'],ascending=False, inplace=True)
    # Checking difference between train and test data accuracy
    MLA_compare['Acc_Diff'] = np.where((np.abs(MLA_compare.MLA_Train_Accuracy - MLA_compare.MLA_Test_Accuracy)) < 0.10, 0, 1)
    MLA_compare['Acc_Diff2'] = MLA_compare.MLA_Train_Accuracy - MLA_compare.MLA_Test_Accuracy

    # Filtering condition
    if MLA_compare[MLA_compare.Acc_Diff == 0].shape[0] > 0:
        MLA_compare = MLA_compare[(MLA_compare.Acc_Diff == 0)]
        MLA_compare = MLA_compare.sort_values(['Acc_Diff2'], ascending=True).head(2)
    else:
        MLA_compare = MLA_compare
        MLA_compare = MLA_compare.sort_values(['Acc_Diff2'], ascending=True).head(2)

    # Dropping columns
    MLA_compare = MLA_compare[MLA_compare.columns.difference(['Acc_Diff', 'Acc_Diff2'])].reset_index(drop=True)
    MLA_compare['Acc_Rank'] = MLA_compare.MLA_Train_Accuracy - MLA_compare.MLA_Test_Accuracy
    MLA_compare["Acc_Rank"] = MLA_compare["Acc_Rank"].rank(method='min')
    MLA_compare = MLA_compare[MLA_compare.Acc_Rank < 5]
    MLA_compare = MLA_compare[['MLA_Name', 'MLA_Train_Accuracy','MLA_Test_Accuracy', 'MLA_Train_Mse', 'MLA_Test_Mse']]

    # Update 'Selection Model'(process_log table)
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS_LOG'], (1,abf.current_date_time(), "Model Selection", process_id_var))
    # Updating Process Table
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS'],(3, 'Running', abf.current_date_time(), process_id))
    logger.info("Model Algorithm Selection is Completed")
  # ==================================================================
  # Hyper Parameter Tuning :
  # ==================================================================
    logger.info("Hyper Parameter Tuning is Started")
    # Hyper Parameter Tuning started(process_log table)
    abf.mysql_execute(mysql_connection,analytics_query_config['INSERT_PROCESS_LOG'], (abf.current_date_time(), "Analytics Layer", 0, "Hyper Parameter Tuning", abf.current_date_time(), process_id))

    # Defining Tuning Parameters
    param_dt = {'max_depth': np.arange(2, 5)}
    param_bg = {'n_estimators': [50, 100, 125, 150]}
    param_rf = {'n_estimators': [50, 100, 125, 150], 'max_depth': [3, 5, 7, 10]}
    param_ada = {'n_estimators': [50, 100, 125, 150],'learning_rate': [10 ** x for x in range(-1, 2)]}
    param_gb = {'n_estimators': [50, 100, 125, 150], 'max_depth': [3, 5, 7, 10], 'learning_rate': [10 ** x for x in range(-1, 2)]}
    param_xg = {'n_estimators': [50, 100, 125, 150], 'max_depth': [3, 5, 7, 10]}
    param_knn = [{'n_neighbors': [3, 7, 9, 11], 'leaf_size':[20, 30, 40, 50]}]
    param_lgb = {'n_estimators': [50, 75, 100,150, 200], 'max_depth': [1, 5, 7, 10]}

    # Model running
    SMLA_columns = []
    SMLA_compare = pd.DataFrame(columns=SMLA_columns)
    row_index = 0

    for i in MLA_compare.MLA_Name:
        if i == 'GradientBoostingRegressor':
            alg = GridSearchCV(estimator=ensemble.GradientBoostingRegressor(), param_grid=param_gb, cv=3, n_jobs=-1)
            alg = alg.fit(TRAIN_X, TRAIN_Y.values.ravel())
            param_gb = alg.best_params_
            model_gb = alg
        elif i == 'XGBRegressor':
            alg = GridSearchCV(estimator=xgboost.XGBRegressor(objective='reg:squarederror'), param_grid=param_xg, cv=3, n_jobs=-1)
            alg = alg.fit(TRAIN_X, TRAIN_Y.values.ravel())
            param_xg = alg.best_params_
            model_xg = alg
        elif i == 'AdaBoostRegressor':
            alg = GridSearchCV(estimator=ensemble.AdaBoostRegressor(), param_grid=param_ada, cv=3, n_jobs=-1)
            alg = alg.fit(TRAIN_X, TRAIN_Y.values.ravel())
            param_ada = alg.best_params_
            model_ada = alg
        elif i == 'RandomForestRegressor':
            alg = GridSearchCV(estimator=ensemble.RandomForestRegressor(), param_grid=param_rf, cv=3, n_jobs=-1)
            alg = alg.fit(TRAIN_X, TRAIN_Y.values.ravel())
            param_rf = alg.best_params_
            model_rf = alg
        elif i == 'BaggingRegressor':
            alg = GridSearchCV(estimator=ensemble.BaggingRegressor(), param_grid=param_bg, cv=3, n_jobs=-1)
            alg = alg.fit(TRAIN_X, TRAIN_Y.values.ravel())
            param_bg = alg.best_params_
            model_bg = alg
        elif i == 'KNeighborsRegressor':
            alg = GridSearchCV(estimator=neighbors.KNeighborsRegressor(), param_grid=param_knn, cv=3, n_jobs=-1)
            alg = alg.fit(TRAIN_X, TRAIN_Y.values.ravel())
            param_knn = alg.best_params_
            model_knn = alg
        elif i == 'DecisionTreeRegressor':
            alg = GridSearchCV(estimator=tree.DecisionTreeRegressor(), param_grid=param_dt, cv=3, n_jobs=-1)
            alg = alg.fit(TRAIN_X, TRAIN_Y.values.ravel())
            param_dt = alg.best_params_
            model_dt = alg
        elif i == 'LGBMRegressor':
            alg = GridSearchCV(estimator=lgb.LGBMRegressor(),param_grid=param_lgb, cv=3, n_jobs=-1)
            alg = alg.fit(TRAIN_X, TRAIN_Y.values.ravel())
            param_lgb = alg.best_params_
            model_lgb = alg
        else:
            alg = linear_model.LinearRegression()
            alg = alg.fit(TRAIN_X, TRAIN_Y.values.ravel())
            param_lin = None
            model_lin = alg

        predicted_train = alg.predict(TRAIN_X)
        predicted_test = alg.predict(TEST_X)
        SMLA_compare.loc[row_index, 'SMLA_Name'] = str(i)
        SMLA_compare.loc[row_index, 'SMLA_Train_Accuracy'] = round(alg.score(TRAIN_X, TRAIN_Y), 4)
        SMLA_compare.loc[row_index, 'SMLA_Test_Accuracy'] = round(alg.score(TEST_X, TEST_Y), 4)
        SMLA_compare.loc[row_index, 'SMLA_Train_Mse'] = metrics.mean_squared_error(TRAIN_Y, predicted_train)
        SMLA_compare.loc[row_index, 'SMLA_Test_Mse'] = metrics.mean_squared_error(TEST_Y, predicted_test)
        row_index += 1

    # Filtering Algorithms based on Accuracy
    Filter_Alg = SMLA_compare[SMLA_compare.SMLA_Train_Accuracy > 0.50]
    if Filter_Alg.shape[0] > 0:
        SMLA_compare = Filter_Alg
    else:
        SMLA_compare = SMLA_compare

    # Calculating Metrics for all the models
    SMLA_compare['MSE_Diff'] = np.abs(SMLA_compare.SMLA_Test_Mse - SMLA_compare.SMLA_Train_Mse)
    SMLA_compare['AVG_Acc'] = (1 - (SMLA_compare.SMLA_Test_Accuracy + SMLA_compare.SMLA_Train_Accuracy) / 2)
    SMLA_compare['ACC_Diff'] = np.abs(SMLA_compare.SMLA_Test_Accuracy - SMLA_compare.SMLA_Train_Accuracy)
    SMLA_compare["MSE_Rank"] = SMLA_compare["MSE_Diff"].rank(method='min')
    SMLA_compare["ACC_Rank"] = SMLA_compare["ACC_Diff"].rank(method='min')
    SMLA_compare['AVG_Rank'] = SMLA_compare['AVG_Acc'].rank(method='max')
    SMLA_compare['Accuracy'] = (SMLA_compare.ACC_Rank + SMLA_compare.MSE_Rank + SMLA_compare.AVG_Rank)

    # Selecting Algorithm based on Accuracy :
    largest = SMLA_compare[SMLA_compare.Accuracy ==SMLA_compare.Accuracy.min()]['SMLA_Name']
    largest = list(largest)
    largest = largest[0]

    # Update 'Hypertuning Parameter completed'(process_log table)
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS_LOG'], (1, abf.current_date_time(), "Hyper Parameter Tuning", process_id))
    # Updating Process Table
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS'],(4, 'Running', abf.current_date_time(), process_id))
    logger.info("Hyper Parameter Tuning is Completed")

  # ==================================================================
  # Model Training :
  # ==================================================================
    logger.info("Model Training is Started")
    # Model Training update (process_log table)
    abf.mysql_execute(mysql_connection,analytics_query_config['INSERT_PROCESS_LOG'], (abf.current_date_time(), "Analytics Layer", 0, "Model Training", abf.current_date_time(), process_id))

    # Building selected model
    if largest == 'GradientBoostingRegressor':
        model = model_gb
        model_param_val = param_gb
    elif largest == 'XGBRegressor':
        model = model_xg
        model_param_val = param_xg
    elif largest == 'AdaBoostRegressor':
        model = model_ada
        model_param_val = param_ada
    elif largest == 'LinearRegression':
        model = model_lin
        model_param_val = param_lin
    elif largest == 'RandomForestRegressor':
        model = model_rf
        model_param_val = param_rf
    elif largest == 'BaggingRegressor':
        model = model_bg
        model_param_val = param_bg
    elif largest == 'KNeighborsRegressor':
        model = model_knn
        model_param_val = param_knn
    elif largest == 'LGBMRegressor':
        model = model_lgb
        model_param_val = param_lgb
    else:
        model = model_dt
        model_param_val = param_dt

    # Saving Model:
    filename = "/Finalized_Model_LTV.sav"
    pickle.dump(model, open(dirpath+filename, 'wb'))
    # Update Model Training(process_log table)
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS_LOG'],(1, abf.current_date_time(), "Model Training", process_id))
    # Updating Process Table
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS'],(5, 'Running', abf.current_date_time(), process_id))
    logger.info("Model Training is Completed")
  # ==================================================================
  # Model Testing :
  # ==================================================================
    logger.info("Model Testing is Started")
    # Model Testing update(process_log table)
    abf.mysql_execute(mysql_connection,analytics_query_config['INSERT_PROCESS_LOG'], (abf.current_date_time(), "Analytics Layer", 0, "Model Testing", abf.current_date_time(), process_id))

    # Storing Prediction results
    TRAIN['Pred_LTV'] = model.predict(TRAIN_X)
    TEST['Pred_LTV'] = model.predict(TEST_X)
    # Checking accuracy for train and test data
    Train_Accuracy = model.score(TRAIN_X, TRAIN_Y)
    TRAIN_Accuracy = format(float(100*Train_Accuracy), '.2f')
    Test_Accuracy = np.abs(model.score(TEST_X, TEST_Y))
    TEST_Accuracy = format(float(100*Test_Accuracy), '.2f')

    logger.info(model.score(TRAIN_X, TRAIN_Y))
    logger.info(model.score(TEST_X, TEST_Y))

    # Checking Mean Square Error for train and test data
    TRAIN_RMSE = int(np.sqrt(metrics.mean_squared_error(TRAIN_Y, TRAIN.Pred_LTV)))
    TEST_RMSE = int(np.sqrt(metrics.mean_squared_error(TEST_Y, TEST.Pred_LTV)))
    # Checking Mean Absolute Error for train and test data
    TRAIN_MAE = int(metrics.mean_absolute_error(TRAIN_Y, TRAIN.Pred_LTV))
    TEST_MAE = int(metrics.mean_absolute_error(TEST_Y, TEST.Pred_LTV))

    # Storing Time Period value in model class metric table
    logger.info("Storing Time Period value in model class metric table")
    abf.mysql_execute(mysql_connection,analytics_query_config['INSERT_RESULTS_IN_CLASS_METRICS'], ("Data", "Time Period", int(TP), int(model_id), abf.current_date_time(), abf.current_date_time()))

    # Storing metrics values for train and test data sets
    logger.info("Storing metrics values for train and test data sets")
    for (i, j) in zip(['Accuracy', 'RMSE', 'MAE'], [TRAIN_Accuracy, TRAIN_RMSE, TRAIN_MAE]):
        abf.mysql_execute(mysql_connection,analytics_query_config['INSERT_RESULTS_IN_CLASS_METRICS'], ("Train", i, j, model_id, abf.current_date_time(), abf.current_date_time()))
    for (i, j) in zip(['Accuracy', 'RMSE', 'MAE'], [TEST_Accuracy, TEST_RMSE, TEST_MAE]):
        abf.mysql_execute(mysql_connection,analytics_query_config['INSERT_RESULTS_IN_CLASS_METRICS'], ("Test", i, j, model_id, abf.current_date_time(), abf.current_date_time()))

    # reading model description using model_id
    description = abf.mysql_fetch(mysql_connection,analytics_query_config['MODEL_DESCRIPTION_FETCH'], (int(model_id),))

    if len(description.strip()) > 0:
        description = description
    else:
        description = "LTV Model"

    # Fetching client_name from client table
    client_name = abf.mysql_fetch(mysql_connection,analytics_query_config['CLIENT_NAME_FETCH'], (client_id,))

    # Updating results model table
    logger.info("Updating results model table")
    abf.mysql_execute(mysql_connection,analytics_query_config['STORING_RESULTS_IN_MODEL_TABLE']['LTV'], (Distinct_Player_Count, format(float(100*Test_Accuracy), '.2f'), str(description),largest, TEST.shape[0], TRAIN.shape[0], abf.current_date_time(), 1, dirpath, "Done", int(client_id), int(user_id), AVG_NGR_VALUE, int(model_id)))

     # Inserting record in model_parameters table
    model_param_val = json.dumps(model_param_val, default=abf.myconverter)
    abf.mysql_execute(mysql_connection, analytics_query_config['INSERT_MODEL_PARAMETERS_TABLE'], (skin_code, abf.current_date_time(), False, largest, model_param_val, abf.current_date_time(), int(client_id), int(model_id), int(model_type)))
    
    # Saving Results
    Test = TEST[['Customer_Id', train_y.name, 'Pred_LTV', 'Segment']]
    Train = TRAIN[['Customer_Id', train_y.name, 'Pred_LTV', 'Segment']]
    Train['data_identifier'] = "Train"
    Test['data_identifier'] = "Test"
    Model_Results = pd.concat([Test, Train], axis=0)
    Model_Results['Client_Id'] = client_id
    Model_Results['User_Id'] = user_id
    Model_Results['Process_Id'] = process_id
    Model_Results = Model_Results[['Customer_Id', 'data_identifier', 'Segment','Pred_LTV', train_y.name, 'Client_Id', 'User_Id', 'Process_Id']]
    Model_Results['Pred_LTV'] = Model_Results.Pred_LTV.astype(int)
    Model_Results.to_excel(dirpath+"/PLTV_Prediction.xlsx")
    Model_Results['create_timestamp'] = abf.current_date_time()
    Model_Results['update_timestamp'] = abf.current_date_time()
    
    #Uploading model results to result table
    MODEL_RESULTS = Model_Results[['Customer_Id', 'data_identifier', 'Segment', 'Pred_LTV', 'Client_Id', 'User_Id', 'Process_Id', 'create_timestamp', 'update_timestamp']]
    abf.upload_results(MODEL_RESULTS,MODEL_RESULTS.shape[1],conn_info,schema_set_query,analytics_schema, analytics_query_config['STORING_CREATION_RESULTS']['LTV'])
    logger.info("Model results upload is finished")

    # ==========================================================================================
    # Comparing Actual and Predicted Values
    # ==========================================================================================
    # Renaming segments
    Train['Segment'] = Train['Segment'].astype(int)
    Train['segment'] = np.where(Train.Segment == 0, "High Value- High Activity", np.where(Train.Segment == 1,'High Value - Medium Activity', np.where(Train.Segment == 2, 'Medium Value - Low Activity', 'Low Value - Low Activity')))
    Test['Segment'] = Test['Segment'].astype(int)
    Test['segment'] = np.where(Test.Segment == 0, "High Value- High Activity", np.where(Test.Segment == 1,'High Value - Medium Activity', np.where(Test.Segment == 2, 'Medium Value - Low Activity', 'Low Value - Low Activity')))

    # Comparing Actual and Predicted values at K-Means Segment level
    apf.seg_act_vs_pre_plt(Train, train_y.name, dirpath+'/', 'Train_Segment_level_comparison.png', f'{target_var.upper()} Value')

    # Comparing Actual and Predicted values at K-Means Segment level
    apf.seg_act_vs_pre_plt(Test, train_y.name, dirpath+'/', 'Test_Segment_level_comparison.png', f'{target_var.upper()} Value')

    # ==========================================================================================
    # Error Plots
    # ==========================================================================================
    # Calling below function to plot error distribution for train data
    apf.err_dist_plt(Train,train_y.name,dirpath+'/','Train_Error_Distribution.png')
    
    # Calling below function to plot error distribution for test data
    apf.err_dist_plt(Test,train_y.name,dirpath+'/','Test_Error_Distribution.png')

    # Residual comparison plot.(Train Data)
    apf.err_comparison_plt(Train,dirpath+'/','Train_Segment_Level_Error_Distribution.png', train_y.name)

    # Residual comparison plot.(Test Data)
    apf.err_comparison_plt(Test,dirpath+'/','Test_Segment_Level_Error_Distribution.png', train_y.name)

    # Connecting to Vertica DB
    vertica_connection = vertica_python.connect(**conn_info)
    vsql_cur = vertica_connection.cursor()
    # Fetching run_id
    vsql_cur.execute(schema_set_query, {'schema_name': analytics_schema})
    vsql_cur.execute(analytics_query_config['RUN_ID_FETCH'])
    insert_run_id = vsql_cur.fetchone()[0]

    # Updating model_feature_engg_run table
    insert_tuple_1 = (insert_run_id, "LTV", int(TP), 0, int(TRAIN.shape[0]), int(TEST.shape[0]), int(Data_out.shape[0]), client_name, model_id, process_id, client_id, start_time, abf.current_date_time(), abf.current_date_time())
    vsql_cur.execute(schema_set_query, {'schema_name': analytics_schema})
    vsql_cur.execute(analytics_query_config['INSERT_FEATURE_ENG_RUN'], insert_tuple_1)
    vertica_connection.commit()

    # Calculating Adjusted R2 value:
    try:
        Adj_R2_value = np.round(1-(1-np.round(Train_Accuracy, 2))*(TRAIN.shape[0]-1)/(TRAIN.shape[0]-len(Features_Ngr)-1), 2)
        if np.isnan(Adj_R2_value) == True or np.isinf(Adj_R2_value) == True:
            Adj_R2_value = np.round(Train_Accuracy, 2)
    except:
        Adj_R2_value = np.round(Train_Accuracy, 2)
        logger.info(Adj_R2_value)

    # Updating model_feature_engg_results table
    insert_tuple_1 = (insert_run_id, np.round(Train_Accuracy, 2), np.round(Test_Accuracy, 2), TRAIN_RMSE, TEST_RMSE,TRAIN_MAE, TEST_MAE, np.round(Train_Accuracy, 2), Adj_R2_value, model_id, process_id, client_id, abf.current_date_time())
    vsql_cur.execute(schema_set_query, {'schema_name': analytics_schema})
    vsql_cur.execute(analytics_query_config['INSERT_FEATURE_RESULTS_REGRESSION'], insert_tuple_1)
    vertica_connection.commit()

    # Sotring feature Importances in feature_list table
    Val_Feature['ID'] = int(insert_run_id)
    Val_Feature['time'] = abf.current_date_time()
    Val_Feature['normalized_importance'] = np.where(Val_Feature.normalized_importance.isnull(), 0, Val_Feature.normalized_importance)
    Val_Feature = Val_Feature[['ID', 'feature','normalized_importance', 'Check', 'time']]
    lists = Val_Feature.values.tolist()
    with vertica_connection.cursor() as cursor:
        cursor.execute(schema_set_query, {'schema_name': analytics_schema})
        for x in lists:
            cursor.execute(analytics_query_config['INSERT_FEATURE_LIST'], x)
            vertica_connection.commit()
    vertica_connection.close()

    # Updating Model Testing Completed(process table)
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS_LOG'],(1, abf.current_date_time(), "Model Testing", process_id))
    # Updating Process Table
    abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS'],(6, 'Done', abf.current_date_time(), process_id))

    # Updating process_type for back-end process
    if process_mode.lower() == "sftp":
        abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_PROCESS_TYPE_IN_PROCESS'],("Create", abf.current_date_time(), process_id))

    # checking if the process is completed without a warning or not
    def mysql_fetch_2(mysql_connection,config_query, input_values):
        '''
        Function to fetch values from the table 
        Parameters:
        1. config_query : SQL query
        2. input_values : SQL values
        '''
        # Connection details for mysql DB
        mysql_connection = abf.mysql_connection(configs)
        cursor = mysql_connection.cursor(prepared=True)
        cursor.execute(config_query, input_values)
        result = cursor.fetchone()
        if result == None:
            value = 'None'
        else:
            #cursor.execute(config_query, input_values)
            value = result[0]
        return value

    warning = mysql_fetch_2(mysql_connection, analytics_query_config['GET_WARNING'], (process_id,29,))
    # err_type = mysql_fetch_2(mysql_connection,analytics_query_config['FETCH_ERROR_TYPE'],(warning,))
    # ERROR_LOG_ID = mysql_fetch_2(mysql_connection, analytics_query_config['FETCH_ERROR_LOG_ID'], (process_id,warning,))

    # Updating Notification Model Creation Status :
    if warning >= 1: #if the process is completed with a warning and no errors
        # abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_NOTIFICATION_FOR_WARNING'], (abf.current_date_time(), "Done with Warning", 0, 1, 0, 0,ERROR_LOG_ID, process_id))
        # ERROR_LOG_ID = mysql_fetch_2(mysql_connection, analytics_query_config['FETCH_ERROR_LOG_ID'], (process_id))
        logger.info("Model Creation is Completed with warning")
        sqln = noti_query_config['UPDATE_NOTIFICATION'] 
        valn = (abf.current_date_time(),(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"), 0, 1, 37,0, 0, process_id, 15)
        abf.sql_update_notification(mysql_connection,sqln,valn)

    else: #if the process is completed is without a warning and without a error
        # abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_NOTIFICATION'],(abf.current_date_time(), "Done", 0, 1, 0, 0, process_id))
        logger.info("Model Creation is Completed without any warning")
        sqln = noti_query_config['UPDATE_NOTIFICATION']
        valn = (abf.current_date_time(),(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"), 0, 1, 1, 0, 0, process_id, 15)
        abf.sql_update_notification(mysql_connection,sqln,valn)
    logger.info("Model Testing is Completed")
    
    if process_mode.lower() == "sftp" or process_mode != "sftp":
        # inserting record into the notification table with 16 notification_type_id
        # abf.mysql_execute(mysql_connection,analytics_query_config['INSERT_NOTIFICATION'], (abf.current_date_time(), 'Model Creation completed', 0, 16, process_id, 0, 0, abf.current_date_time()))
        sqln = noti_query_config['INSERT_NOTIFICATION']
        valn = (abf.current_date_time(), -1, 1, 1, process_id, 0, 0, abf.current_date_time(), (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),0)
        abf.sql_update_notification(mysql_connection,sqln,valn)
    
    # Removing uploaded files
    logger.info("Removing uploaded is Started")
    abf.remove_upload_files(acc_home,configs,client_id,user_id,process_id,process_mode,logger)
    logger.info("Removing uploaded is Finished")

    # Calling dashboard_enrichment script
    try:
        if process_mode != "sftp":
            abf.dashboard_script_call(acc_home,client_id,user_id,process_id,upload_id,model_type,logger)
    except:
        # abf.mysql_execute(mysql_connection,analytics_query_config['UPDATE_NOTIFICATION'], (abf.current_date_time(), "Incremental data load is failed.", 0, 15, 0, 0, process_id))
        sqln = noti_query_config['INSERT_NOTIFICATION']
        valn = (abf.current_date_time(), -1, 38, 12, process_id, 0, 0, abf.current_date_time(),(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),0)
        abf.sql_update_notification(mysql_connection,sqln,valn)        
        sys.exit()
