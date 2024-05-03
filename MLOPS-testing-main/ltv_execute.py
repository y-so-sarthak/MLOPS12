# LTV Model Execution Script :
def execute_model(client_id, user_id, process_id, upload_id, model_id, model_type, process_mode, drift_process_yn=0, threshold_val=0.2, qcut_yn=None):
    # ==================================================================
    # Importing Packages
    # ==================================================================
    import os
    import matplotlib
    matplotlib.use('Agg')
    import pickle
    import numpy as np
    import pandas as pd
    import json
    import sys
    import shutil
    import vertica_python
    from io import StringIO
    from datetime import datetime, timedelta
    import matplotlib.ticker as ticker
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import analytics_model_functions as amf
    import analytics_base_functions as abf
    import warnings
    warnings.filterwarnings('ignore')
    import ltv_impact_analysis as lia
    import plotly.io as pio
    scope = pio.kaleido.scope
 

    def current_date_time():
        '''Function for fetching current date and time.'''
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # ==================================================================
    # Connecting with DB
    # ==================================================================
    # Reading environment variable
    acc_home = os.environ['ACCELERATOR_HOME']
    acc_home = acc_home+'/'
    with open(acc_home+'python_scripts/config.json', 'r') as json_file:
        configs = json.load(json_file)
    with open(acc_home + 'python_scripts/client_config.json', 'r') as json_file:
        client_configs = json.load(json_file)
    with open(acc_home+'python_scripts/notification_query.json', 'r') as n:
        noti_query_config = json.load(n)
    # Directory for fetching log file.
    logger = abf.log_file_initiate(acc_home,configs,client_id,user_id,process_id)
    logger.info("Logger initialized successfully.")

    # Reading my sql scripts
    with open(acc_home+'python_scripts/analytics_stage_query.json', 'r') as json_file:
        analytics_query_config = json.load(json_file)
    with open(acc_home+'sql_scripts/vertica_schema_set.sql', 'r') as sql_file:
        schema_set_query = sql_file.read()
    schema_qualifier = os.environ['SCHEMA_QUALIFIER']

    # using the env variable to switch to dev schemas here
    schema_type = configs['schema_type'][schema_qualifier]
    analytics_schema = schema_type['analytics']
    # =>Connection Details:
    conn_info = abf.vertica_connection(configs)

    # Connecting to the DB
    vertica_connection = vertica_python.connect(**conn_info)
    vsql_cur = vertica_connection.cursor()
    logger.info("Successfully connected to Vertica DB")
    # Query for extracting data to particular Process_Id
    process_id_var = str(process_id)
    client_id_var = str(client_id)
    # Executing the statement
    vsql_cur.execute(schema_set_query, {'schema_name': analytics_schema})
    try:
        client_model = client_configs.get('client').get(f'c{int(client_id_var)}').get('client_model', None)
    except:
        client_model = None
    if client_model == 'OLD':
        vsql_cur.execute(analytics_query_config['DATA_QUERY']['LTV_OLD'], {
            'process_id_var': process_id_var, 'client_id_var': client_id_var})
    else:
        vsql_cur.execute(analytics_query_config['DATA_QUERY']['LTV'], {
            'process_id_var': process_id_var, 'client_id_var': client_id_var})
    # Converting into DataFrame formate
    Validation = pd.DataFrame(vsql_cur.fetchall())
    logger.info("Fetching data from Vertica DB is Completed")
    Validation.columns = [c.name for c in vsql_cur.description]
    vertica_connection.close()

    # Connection detils for MySQL DB
    mysql_connection = abf.mysql_connection(configs)
    logger.info("Successfully connected to MySQL DB")

    logger.info(Validation.columns)
    logger.info(Validation.shape)

    # reading out the model_execute_id from process table
    model_execution_id = abf.mysql_fetch(mysql_connection, analytics_query_config['MODEL_EXECUTION_FETCH'], (process_id,))

    # reading out the graph path using model_id
    model_path = abf.mysql_fetch(mysql_connection, analytics_query_config['GRAPH_PATH_FETCH'], (model_id,))
    model_path = model_path + '/'

    # fetching target variable name from model table
    ltv_target = abf.mysql_fetch(mysql_connection, analytics_query_config['TARGET_VAR_FETCH'], (model_id,))
    rfm_seg_col = ltv_target.capitalize() + '_Overall'
    Validation.rename(columns={"actual_ngr":ltv_target},inplace=True)

    # Brand_name and Prediction date fetch from process table
    if process_mode != "sftp":
        brand_name = str(abf.mysql_fetch(mysql_connection, analytics_query_config['BRAND_NAME_FETCH_MODEL'], (model_id,)))
        logger.info(brand_name)
        if brand_name == "None":
            brand_name = 'all'
        prediction_date_yn = abf.mysql_fetch(mysql_connection, analytics_query_config['PREDICTION_DATE_YN'],(process_id,))
        logger.info(prediction_date_yn)
        if int(prediction_date_yn) == 1:
            end_date = abf.mysql_fetch(mysql_connection, analytics_query_config['MAX_DATE_FETCH_PROCESS'], (process_id,))
            logger.info(end_date)
        else:
            end_date = abf.mysql_fetch(mysql_connection, analytics_query_config['MAX_DATA_DATE_FETCH_PROCESS'], (upload_id,))
            logger.info(end_date)
    else:
        brand_name = str(abf.mysql_fetch(mysql_connection, analytics_query_config['BRAND_NAME_FETCH_MODEL'], (model_id,)))
        logger.info(brand_name)
        if brand_name == "None":
            brand_name = 'all'
        end_date = abf.mysql_fetch(mysql_connection, analytics_query_config['MAX_DATE_FETCH_PROCESS'], (process_id,))
        logger.info(end_date)


    # No Samples in training model
    if int(Validation.shape[0]) < 1:
        logger.info("No Data for Model Execution")
        abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_ERROR_IN_MODEL_EXECUTION_TABLE'], ("Error", abf.current_date_time(), model_execution_id))
        sys.exit()

    # reading out time period using model_id
    TP = Validation['prediction_period'].unique()[0]
    logger.info(TP)
    # Setting Model Path
    # ==================================================================
    # Data Preparation :
    # ==================================================================
    logger.info("Data Preparation is Started")
    # Updating Process Table & Inserting Process Log Table Based on Drift Process Value..
    if drift_process_yn == 1:
        abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS_FIRST'], (0, 2, "Data Drift", abf.current_date_time(), process_id))
        abf.mysql_execute(mysql_connection, analytics_query_config['INSERT_PROCESS_LOG'], (abf.current_date_time(), "Data Drift", 0, "Data Preparation", abf.current_date_time(), process_id))
    else:
        abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS_FIRST'], (0, 3, "Analytics", abf.current_date_time(), process_id))
        abf.mysql_execute(mysql_connection, analytics_query_config['INSERT_PROCESS_LOG'], (abf.current_date_time(), "Analytics Layer", 0, "Data Preparation", abf.current_date_time(), process_id))

    # No Samples in training model
    if Validation.shape[0] < 0:
        logger.info("No Data for Model Execution")
        abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_ERROR_IN_MODEL_EXECUTION_TABLE'], ("Error", abf.current_date_time(), model_execution_id))
        sys.exit()

    # More Missing values in test data.
    if Validation.isnull().sum().sum() > (int(Validation.shape[0]*Validation.shape[1]*0.35)):
        logger.info("More number of Missing Values")

    # Dropping Variables which have more than 35% missing values
    Drop_Var = pd.read_csv(model_path+"Droped_Var.csv")

    # Dropping variable list
    if Drop_Var.shape[0] > 0:
        Validation.drop(list(Drop_Var.Columns), axis=1, inplace=True)

    # Seperating Numerical and categorical data type column names
    try:
        # Fetching numerical and categorical columns from model directory path
        with open(model_path+"Features_Num.bin", "rb") as data:
            num_col = pickle.load(data)
        with open(model_path+"Features_Cat.bin", "rb") as data:
            cat_col = pickle.load(data)

        # Seperating Numerical and categorical variables
        Validation_Num = Validation[num_col]
        Validation_Cat = Validation[cat_col]
        logger.info("Seperated numerical and categorical columns")
    except:
        logger.info("Seperating numerical and Categorical columns based on columns data types")
        # Seperating numerical and categorical columns based on thier data types
        num_col = [key for key in dict(Validation.dtypes) if dict(Validation.dtypes)[key] in ['int64', 'int32', 'float64', 'float32']]
        cat_col = [key for key in dict(Validation.dtypes) if dict(Validation.dtypes)[key] in ['object']]

        # Seperating Numerical and categorical variables data
        Validation_Num = Validation[num_col]
        Validation_Cat = Validation[cat_col]
        logger.info("Seperated numerical and categorical columns using their data types..")

    # Imputing missing values:
    loaded_missing = pickle.load(open(model_path+'Missing_Impution_Object.sav', 'rb'))
    Validation_Num = pd.DataFrame(loaded_missing.transform(Validation_Num[Validation_Num.columns]), columns=Validation_Num.columns)
    Validation_Num.index = Validation_Cat.index

    # #### Outlier Treatment :
    for col in Validation_Num.columns.difference(['Customer_Id', 'prediction_period', 'actual_apd']):
        percentiles = Validation_Num[col].quantile([0.01, 0.99]).values
        Validation_Num[col] = np.clip(Validation_Num[col], percentiles[0], percentiles[1])

    # Reading Categories from Categorical table
    Categories = pd.read_csv(model_path+"Categories_LTV.csv")
    Categories = Categories.iloc[:, 1:]

    # Reducing Categories
    try:
        for i in Categories.columns:
            Val = list(Categories[str(i)].values)
            Validation_Cat[str(i)] = np.where(Validation_Cat[str(i)] == Val[0], Val[0],
                                              np.where(Validation_Cat[str(i)] == Val[1], Val[1],
                                                       np.where(Validation_Cat[str(i)] == Val[2], Val[2], 'Others')))
    except ValueError:
        # Updating AL_8 error in error_log table
        abf.mysql_execute(mysql_connection, analytics_query_config['INSERT_ERROR_LOG'], (abf.current_date_time(), "AL_8", process_id))
        # Updating error in notification table about :
        # mysql_execute(analytics_query_config['UPDATE_NOTIFICATION'], (current_date_time(), "Error", 0, 2, 0, 0, process_id))
        sqln = noti_query_config['UPDATE_NOTIFICATION']
        valn = (abf.current_date_time(),(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"), 0, 15, 28, 0, 0, process_id, 15)
        abf.sql_update_notification(mysql_connection,sqln,valn)
        # Updating Error in Model Execution table
        abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_ERROR_IN_MODEL_EXECUTION_TABLE'], ("Error", abf.current_date_time(), model_execution_id))
        sys.exit()

    # Imputing Missing values with Mode Value
    for i in Validation_Cat.columns.difference(['Customer_Id']):
        Validation_Cat[str(i)].fillna(Validation_Cat[str(i)].mode()[0], inplace=True)

    logger.info("Missing value imputation done")

    # Creating dummy variable for categorical variables
    for c_feature in Validation_Cat.columns.difference(['Customer_Id']):
        Validation_Cat[c_feature] = Validation_Cat[c_feature].astype('category')
        Validation_Cat = abf.create_dummies(Validation_Cat, c_feature)

    logger.info("dummary variable creation finished")
    # Combining both numerical and categorical data
    Validation_Num.reset_index(drop=True, inplace=True)
    Validation_Cat.reset_index(drop=True, inplace=True)
    Validation = pd.concat([Validation_Num, Validation_Cat], axis=1)
    # Symbols are not considered in Column names. Replacing with 2
    Validation.columns = Validation.columns.str.strip()
    Validation.columns = Validation.columns.str.replace(' ', '_')
    Validation.columns = Validation.columns.str.replace(r"[^a-zA-Z\d\_]+", "")
    Validation.columns = Validation.columns.str.replace(r"[^a-zA-Z\d\_]+", "")
    logger.info("Data Preparation is Completed")

    # Creating path for storing execute results
    root = acc_home+configs['home']['result']
    dirname = "{}/{}/{}/".format(client_id, user_id, process_id)
    r_dirpath = os.path.join(root, dirname)

    # =======================================================
    #  ### segmentation
    # =======================================================
    # Calculating Quantiles for variables
    quantiles = Validation[['Days_Since_Last_Bet', 'Apd_Overall', rfm_seg_col]].quantile(q=[0.25, 0.5, 0.75])
    quantiles = quantiles.to_dict()
    filename = "RFM_Quantile_LTV.sav"
    pickle.dump(quantiles, open(r_dirpath+filename, 'wb'))
    # User defined function for Creating segments using Recency Variable
    # Arguments (x = value, p = recency, frequency,monetary_value, k = quartiles dict)


    # Implementing user defined function for creating segments.
    Validation['R_Quartile'] = Validation['Days_Since_Last_Bet'].apply(amf.RClass, args=('Days_Since_Last_Bet', quantiles,))
    Validation['F_Quartile'] = Validation['Apd_Overall'].apply(amf.FClass, args=('Apd_Overall', quantiles,))
    Validation['M_Quartile'] = Validation[rfm_seg_col].apply(amf.MClass, args=(rfm_seg_col, quantiles,))
    # Overall Quartile RFM Segments
    Validation['Overall_RFMQuartile_Score'] = Validation['R_Quartile'] + Validation['F_Quartile'] + Validation['M_Quartile']
    # Creating aggregated by FRM Quantile score
    DATA_Q_Rank = Validation.groupby('Overall_RFMQuartile_Score', as_index=False).agg({"Days_Since_Last_Bet": "mean", "Apd_Overall": "mean", rfm_seg_col: 'mean', "Customer_Id": "count"})
    # Calculating M Quantiles for variables
    M_quantiles = DATA_Q_Rank[[rfm_seg_col]].quantile(q=[0.25, 0.5, 0.75])
    M_quantiles = M_quantiles.to_dict()
    # Saving M_Qunatile
    filename = "M_Quantile_LTV.sav"
    pickle.dump(M_quantiles, open(r_dirpath+filename, 'wb'))

    # Creating segment based on qunatile score
    DATA_Q_Rank['Segment'] = 0
    DATA_Q_Rank.loc[DATA_Q_Rank[rfm_seg_col] <= DATA_Q_Rank[rfm_seg_col].quantile(0.75), 'Segment'] = 1
    DATA_Q_Rank.loc[DATA_Q_Rank[rfm_seg_col] <= DATA_Q_Rank[rfm_seg_col].quantile(0.50), 'Segment'] = 2
    DATA_Q_Rank.loc[DATA_Q_Rank[rfm_seg_col] <= DATA_Q_Rank[rfm_seg_col].quantile(0.25), 'Segment'] = 3
    Validation = pd.merge(Validation, DATA_Q_Rank[["Overall_RFMQuartile_Score", "Segment"]], how="left", on=["Overall_RFMQuartile_Score"])
    logger.info("Player Segmentation is Completed")

    # Reading All selected variables for three target variables
    with open(model_path+"Features_Ngr.bin", "rb") as data:
        Features_Ngr = pickle.load(data)

    # Reading all feature variables
    with open(model_path+"All_Features_LTV.bin", "rb") as data:
        All_Features_LTV = pickle.load(data)

    # Reading Top-5 features
    try:
        with open(model_path+"Features_Ngr_5.bin", "rb") as data:
            key_feature_list = pickle.load(data)
        key_feature_list.append('Customer_Id')
        key_feature_yn = 1
    except:
        logger.info("Top-5 feature file not found")
        key_feature_yn = 0

    # Checking selected for NGR Model
    for i in Features_Ngr:
        if i not in list(Validation.columns):
            Validation[str(i)] = 0

    if drift_process_yn == 1:
        # Update Process & Process Log Tables
        abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS_LOG'], (1, abf.current_date_time(), "Data Preparation", process_id))
        abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS'], (1, 'Running', abf.current_date_time(), process_id))
        # Inserting Data Drift sub stage in process_log table.
        abf.mysql_execute(mysql_connection, analytics_query_config['INSERT_PROCESS_LOG'], (abf.current_date_time(), "Data Drift", 0, "Drift Calculation", abf.current_date_time(), process_id))

    # Filtering data with selected features
    Test_x = Validation[Features_Ngr]

    Test_x_drift = Validation[All_Features_LTV]

    # Checking Key Features Data Drift values...
    try:
        amf.data_drift_check(Test_x_drift, model_id, process_id, client_id, str(end_date), analytics_query_config, conn_info, schema_set_query, analytics_schema, threshold_val=0.2, qcut_yn=None)
    except Exception as ex:
        logger.info(ex)
        if drift_process_yn == 1:
            # Updating Error in error_log & notification tables
            abf.mysql_execute(mysql_connection, analytics_query_config['INSERT_ERROR_LOG'],(abf.current_date_time(), "DD_1", process_id))
            # abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_NOTIFICATION'], (abf.current_date_time(), "Error", 0, 2, 0, 0, process_id))
            sqln = noti_query_config['UPDATE_NOTIFICATION']
            valn = (abf.current_date_time(), (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"), 0, 15, 28, 0, 0, process_id, 15)
            abf.sql_update_notification(mysql_connection,sqln,valn)
            sys.exit()

    # Breaking process if it is a data drift process.
    if drift_process_yn == 1:
        # Updating Process Status in Process, Process_Log, and Notification Tables
        abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS'], (2, 'Done', abf.current_date_time(), process_id))
        abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS_LOG'], (1, abf.current_date_time(), "Drift Calculation", process_id))
        # abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_NOTIFICATION'], (abf.current_date_time(), "Done", 0, 1, 0, 0, process_id))
        sqln = noti_query_config['UPDATE_NOTIFICATION']
        valn = (abf.current_date_time(), (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"), 0, 1, 2, 0, 0,process_id, 15)
        abf.sql_update_notification(mysql_connection,sqln,valn)
        sys.exit()

    # Update 'Data Preparation completed'(process_log table)
    abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS_LOG'], (1, abf.current_date_time(), "Data Preparation", process_id))
    # Updating Process Table
    abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS'], (1, 'Running', abf.current_date_time(), process_id))
  # ==================================================================
  # Loading Model
  # ==================================================================
    # Loading model status in process_log table
    abf.mysql_execute(mysql_connection, analytics_query_config['INSERT_PROCESS_LOG'], (abf.current_date_time(), "Analytics Layer", 0, "Loading Model", abf.current_date_time(), process_id))
    # load the model from disk
    loaded_model = pickle.load(open(model_path+'Finalized_Model_LTV.sav', 'rb'))
    # Update 'Loading Model completed'
    abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS_LOG'], (1, abf.current_date_time(), "Loading Model", process_id))
    # Updating Process Table
    abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS'], (2, 'Running', abf.current_date_time(), process_id))
    logger.info("Loading Model is Completed")
  # ==================================================================
  # Result Prediction
  # ==================================================================
    logger.info("Result Prediction is Started")
    # Update results prediction in process_log table
    abf.mysql_execute(mysql_connection, analytics_query_config['INSERT_PROCESS_LOG'], (abf.current_date_time(), "Analytics Layer", 0, "Result Prediction", abf.current_date_time(), process_id))

    # Filtering top-5 features data
    if key_feature_yn == 1:
        # Seperating Top-5 features data
        key_features_data = Validation[key_feature_list]
        Key_Features_Data = key_features_data[key_features_data.columns.difference(['Customer_Id'])].round(2)
        Key_Features_Data['Customer_Id'] = key_features_data.Customer_Id

    # Predicting values using data
    Predictions = pd.DataFrame()
    Predictions['Customer_Id'] = Validation.Customer_Id
    Predictions['Segment'] = Validation.Segment
    Predictions['Pred_LTV'] = np.round(loaded_model.predict(Test_x), 2)

    # Validating Prediction results
    if Predictions.shape[0] > 25:
        if len(Predictions['Pred_LTV'].value_counts()) < 2:
            # Updating AL_3 error in error_log table
            abf.mysql_execute(mysql_connection, analytics_query_config['INSERT_ERROR_LOG'], (abf.current_date_time(), "AL_3", process_id))
            # Updating error in notification table about :
            # abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_NOTIFICATION'], (abf.current_date_time(), "Error", 0, 2, 0, 0, process_id))
            sqln = noti_query_config['UPDATE_NOTIFICATION']
            valn = (abf.current_date_time(), (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"), 0, 15, 28, 0, 0, process_id, 15)
            abf.sql_update_notification(mysql_connection,sqln,valn)
            # Updating Error in Model Execution table
            abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_ERROR_IN_MODEL_EXECUTION_TABLE'], ("Error", abf.current_date_time(), model_execution_id))
            logger.info("All Predctions values are same")
            Predictions.to_csv(r_dirpath+'Prediction_Error1.csv', index=False)
            sys.exit()
        elif Predictions[(Predictions.Pred_LTV >= 0) & (Predictions.Pred_LTV < 1)].shape[0] == Predictions.shape[0]:
            # Updating AL_3 error in error_log table
            abf.mysql_execute(mysql_connection, analytics_query_config['INSERT_ERROR_LOG'], (abf.current_date_time(), "AL_3", process_id))
            # Updating error in notification table about :
            # abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_NOTIFICATION'], (abf.current_date_time(), "Error", 0, 2, 0, 0, process_id))
            sqln = noti_query_config['UPDATE_NOTIFICATION']
            valn = (abf.current_date_time(), (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"), 0, 15, 28, 0, 0, process_id, 15)
            abf.sql_update_notification(mysql_connection,sqln,valn)
            # Updating Error in Model Execution table
            abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_ERROR_IN_MODEL_EXECUTION_TABLE'], ("Error", abf.current_date_time(), model_execution_id))
            logger.info("All Predictions values are between 0 and 1")
            Predictions.to_csv(r_dirpath+'Prediction_Error2.csv', index=False)
            sys.exit()

    logger.info("Prediction values are looks good.")
    validation_date = end_date + timedelta(days=(int(TP)))
    Predictions['Client_Id'] = client_id
    Predictions['User_Id'] = user_id
    Predictions['Process_Id'] = process_id
    Predictions['LTV_Time_Period'] = int(TP)
    Predictions['Execution_Date'] = datetime.now().strftime("%Y-%m-%d")
    Predictions['Prediction_Date'] = end_date
    Predictions['Validation_Date'] = validation_date
    Predictions['Segment'] = Predictions['Segment'].astype(int)
    Predictions['segment'] = np.where(Predictions.Segment == 0, "High Value- High Activity", np.where(Predictions.Segment == 1,
                                      'High Value - Medium Activity', np.where(Predictions.Segment == 2, 'Medium Value - Low Activity', 'Low Value - Low Activity')))

    # Result file
    Prediction_File = Predictions[['Customer_Id', 'Pred_LTV', 'segment', 'LTV_Time_Period', 'Prediction_Date']]
    Prediction_File.columns = ['Customer_Id', 'Predicted_LTV', 'Segment', 'LTV_Time_Period', 'Prediction_Date']
    Prediction_File['Predicted_LTV'] = np.round(Prediction_File.Predicted_LTV, 2)
    Prediction_File['Brand_Name'] = brand_name
    Prediction_File = Prediction_File[['Customer_Id', 'Predicted_LTV', 'Segment', 'LTV_Time_Period', 'Brand_Name', 'Prediction_Date']]

    # API Response
    Prediction_API = Prediction_File[['Customer_Id', 'Predicted_LTV', 'Segment', 'LTV_Time_Period', 'Brand_Name']]
    Prediction_API['Customer_Id'] = Prediction_API['Customer_Id'].astype(str)
    Prediction_API['Predicted_LTV'] = Prediction_API['Predicted_LTV'].astype(str)
    Prediction_API['Segment'] = Prediction_API['Segment'].astype(str)
    Prediction_API['LTV_Time_Period'] = Prediction_API['LTV_Time_Period'].astype(str)
    Prediction_API['Brand_Name'] = Prediction_API['Brand_Name']

    # Saving Results files with key_feature value
    if key_feature_yn == 1:
        # Removing underscore symbols in fature names.
        Feature_Names = list()
        feature_names = list(Key_Features_Data.columns)
        for i in feature_names:
            if i == 'Customer_Id':
                Feature_Names.append(i)
            else:
                Feature_Names.append(i.replace("_", " "))
        Key_Features_Data.columns = Feature_Names
        # Combining model results and key features
        Prediction_File = pd.merge(Prediction_File, Key_Features_Data, left_on='Customer_Id', right_on='Customer_Id', how='left')
        Prediction_File.to_csv(r_dirpath+"Prediction.csv", index=False)
    else:
        logger.info("Saving prediction results without key_features.")
        Prediction_File.to_csv(r_dirpath+"Prediction.csv", index=False)
    Predictions.drop(['segment'], axis=1, inplace=True)
    Predictions.Pred_LTV = Predictions.Pred_LTV.astype(int)
    Predictions['create_timestamp'] = abf.current_date_time()
    Predictions['update_timestamp'] = abf.current_date_time()
    Predictions['model_id'] = int(model_id)
    Predictions['model_execution_id'] = int(model_execution_id)

    # Storing Results in DB with key_feature value
    if key_feature_yn == 1:
        logger.info("Storing prediction results with key_features.")
        Predictions = pd.merge(Predictions, Key_Features_Data,left_on='Customer_Id', right_on='Customer_Id', how='left')
        Predictions['brand_name'] = brand_name
        # temporary buffer
        buff = StringIO()
        # convert data frame to csv type
        for row in Predictions.values.tolist():
            buff.write('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n'.format(*row))
        # Storing results into vertica table
        vertica_connection = vertica_python.connect(**conn_info)
        with vertica_connection.cursor() as cursor:
            cursor.execute(schema_set_query, {'schema_name': analytics_schema})
            cursor.copy(analytics_query_config['STORING_RESULTS']['LTV_KEY_FEATURE'], buff.getvalue())
            vertica_connection.commit()
        vertica_connection.close()
        logger.info("Storing prediction results in DB is Completed")
    else:
        Predictions['brand_name'] = brand_name
        logger.info("Storing prediction results without key_features.")
        # temporary buffer
        buff = StringIO()
        # convert data frame to csv type
        for row in Predictions.values.tolist():
            buff.write('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n'.format(*row))
        # Storing results into vertica table
        vertica_connection = vertica_python.connect(**conn_info)
        with vertica_connection.cursor() as cursor:
            cursor.execute(schema_set_query, {'schema_name': analytics_schema})
            cursor.copy(analytics_query_config['STORING_RESULTS']['LTV'], buff.getvalue())
            vertica_connection.commit()
        vertica_connection.close()
        logger.info("Storing prediction results in DB is Completed")

    # Converting
    Predictions['segment'] = np.where(Predictions.Segment == 0, "High Value- High Activity", np.where(Predictions.Segment == 1, 'High Value - Medium Activity', np.where(Predictions.Segment == 2, 'Medium Value - Low Activity', 'Low Value - Low Activity')))
    # Plotting Ngr Value at Segment level
    fig = plt.figure(num=None, figsize=(6.5, 6), facecolor='w', edgecolor='k')
    for i, j in zip(range(1, 5), analytics_query_config['LTV_SEGMENTS']):
        plt.subplot(2, 2, i)
        Title = "Segment : " + str(j)
        plt.title(Title, fontweight="bold", loc="left", fontsize="8")
        s = Predictions[Predictions.segment == j]['Pred_LTV']
        plt.hist(s, bins=25, color='#32AAE1')
        plt.xlabel(f"{ltv_target.upper()} Value")
        plt.ylabel("Player Count")
        plt.tight_layout(pad=1.75)
        plt.subplots_adjust(bottom=.125, left=.125)
        plt.xticks(rotation=12)
        plt.yticks(rotation=12)
        plt.gca().xaxis.set_major_formatter(ticker.EngFormatter())
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_formatter(ticker.EngFormatter())
        fig.subplots_adjust(top=0.90)
        fig.suptitle(f'Predicted {ltv_target.upper()} Distribution', fontweight="bold")
        plt.savefig(r_dirpath + "segment_plot.png")
        scope._shutdown_kaleido()

    # Converting into json format
    if key_feature_yn == 1:
        try:
            feature_names.remove('Customer_Id')
        except:
            logger.info('Customer_Id is not exist in list')
    else:
        feature_names = list()
    metrics_dict = {"prediction_date": str(end_date), "brand_name": str(brand_name), "Time_Period": int(TP), "key_features": list(feature_names)}
    metrics_dict = json.dumps(metrics_dict)
    # Updating model_execution table
    abf.mysql_execute(mysql_connection, analytics_query_config['MODEL_EXECUTION_UPDATE_QUERY']['LTV'], (model_id, metrics_dict, "Done", int(Test_x.shape[0]), r_dirpath, validation_date, model_execution_id))

    max_date = abf.mysql_fetch(mysql_connection,analytics_query_config['MAX_DATE_FETCH_CLIENT'], (client_id, brand_name,))

    # Update 'Result Prediction completed'
    abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS_LOG'], (1, abf.current_date_time(), "Result Prediction", process_id))
    # Updating Process Table
    abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS'], (3, 'Done', abf.current_date_time(), process_id))

    # Updating process_type for back-end process
    if process_mode == "sftp":
        abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_PROCESS_TYPE_IN_PROCESS'], ("Execute", abf.current_date_time(), process_id))
    # Updating Notification Model Execution Status :
    # abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_NOTIFICATION'], (abf.current_date_time(), "Done", 0, 1, 0, 0, process_id))
    sqln = noti_query_config['UPDATE_NOTIFICATION']
    valn = (abf.current_date_time(),(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"), 0, 1, 2, 0, 0, process_id, 15)
    abf.sql_update_notification(mysql_connection,sqln,valn)
    logger.info("Result Prediction is Completed")

    if process_mode == "sftp":
        # inserting record into the notification table with 16 notification_id
        # abf.mysql_execute(mysql_connection, analytics_query_config['INSERT_NOTIFICATION'], (abf.current_date_time(), 'Model Execution completed', 0, 16, process_id, 0, 0, abf.current_date_time()))
        sqln = noti_query_config['INSERT_NOTIFICATION']
        valn = (abf.current_date_time(), -1, 2, 1, process_id, 1, 0, abf.current_date_time(), (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),0)
        abf.sql_update_notification(mysql_connection,sqln,valn)
    
    # Removing uploaded files
    logger.info("Started Removing Uploaded Files")
    abf.remove_upload_files(acc_home, configs, client_id, user_id, process_id, process_mode, logger)
    logger.info("Finished Removing Uploaded Files")

    logger.info('ltv model execution completed')

    # Calling dashboard_enrichment script
    logger.info("process_mode: %s", process_mode)
    try:
        if process_mode != "sftp":
            logger.info("-1 level data updation for frontend execution")
            Root = acc_home
            Dirname = "python_scripts"
            Dirpath = os.path.join(Root, Dirname)
            logger.info(Dirpath)
            os.chdir(Dirpath)
            from segment_enrichment import Segment_enrichment
            Segment_enrichment(process_id, logger)
            logger.info("Executed segment enrichment script")
            from dashboard_data_enrichment import dashboard_enrichment
            from data_validation import validation
            if process_id == upload_id:
                dashboard_enrichment(process_id, client_id, logger)
                logger.info("Executed dashboard enrichment script")
                logger.info("dw data validation started")
                validation(client_id, user_id, process_id, process_id, process_id, model_type, False, True)
                logger.info('dw data validation completed.')
    except:
        # abf.mysql_execute(mysql_connection, analytics_query_config['UPDATE_NOTIFICATION'], (abf.current_date_time(), "Incremental data load is failed.", 0, 15, 0, 0, process_id))
        sqln = noti_query_config['']
        valn = (abf.current_date_time(), -1, 38, 12, process_id, 0, 0, abf.current_date_time(), (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),0)
        abf.sql_update_notification(mysql_connection,sqln,valn) 
        sys.exit()

    logger.info('prediction_date: %s', end_date)
    logger.info('max_date: %s', max_date)
    logger.info('validation_date: %s', validation_date)
    logger.info("model_execution_id: %s", model_execution_id)
    try:
        logger.info('LTV impact analysis started for this execution')
        flag = 1
        logger.info("flag: %s", flag)
        if validation_date >= max_date:
            logger.info("This is an ACTIVE model execution w.r.t impact analysis")
            if max_date > end_date + timedelta(days=(int(TP/3))):
                logger.info("This model execution has actual data for more than a third of its prediction period")
                data_size = lia.actual_val_update_default(client_id, brand_name, model_id, str(end_date), TP, str(max_date), flag, model_execution_id, logger)
                if (data_size > 0):
                    logger.info("Actual data update complete")
                    test = lia.actual_val_default_data_fetch(model_execution_id, brand_name, client_id, end_date, max_date, logger)
                    logger.info("Test Dataframe: %s", test)
                    lia.impact_analysis(int(model_execution_id), TP, end_date,
                                max_date, conn_info, mysql_connection, analytics_query_config, test, logger)
                else:
                    logger.info("Actual data not present for current execution when max date is less than or equal to validation date")
                    print("InComplete")
            else:
                logger.info("This model execution does not have actual data for more than a third of its prediction period")
                print("InComplete")
        else:
            logger.info("This is an INACTIVE model execution w.r.t impact analysis")
            # flag = 1
            # logger.info("flag: %s", flag)
            data_size = lia.actual_val_update_default(client_id, brand_name, model_id, str(end_date), TP, str(validation_date), flag, model_execution_id, logger)
            if (data_size > 0):
                logger.info("Actual data update complete")
                test = lia.actual_val_default_data_fetch(model_execution_id, brand_name, client_id, end_date, validation_date, logger)
                logger.info("Test Dataframe: %s", test)
                lia.impact_analysis(int(model_execution_id), TP, end_date,
                            validation_date, conn_info, mysql_connection, analytics_query_config, test, logger)
            else:
                logger.info("Actual data not present for current execution when max date is greater than validation date")
                print("InComplete")
        logger.info('LTV impact analysis completed for this execution')
    except Exception as ex:
        logger.info(ex)

    try:
        flag = 0
        logger.info("LTV impact analysis for previous active executions started")
        cursor = mysql_connection.cursor(prepared=True)
        cursor.execute(analytics_query_config['EXECUTION_DATA_FETCH']['LTV'], (str(model_id),str(max_date),str(TP),str(max_date),str(int((TP/3)+1)),))
        default_execs = cursor.fetchall()
        mysql_connection.commit()
        cursor.close()

        logger.info("Previous active executions(model_execution_id, prediction_date): ")
        if len(default_execs) > 0:
            for x in default_execs:
                logger.info(x)
            min_default_execs = min(default_execs, key=lambda x: x[1])
            min_model_execution_id = min_default_execs[0]
            min_prediction_date = min_default_execs[1]
            data_size = lia.actual_val_update_default(client_id, brand_name, model_id, str(min_prediction_date), TP, str(max_date), flag, min_model_execution_id, logger)
            logger.info("data_size for all previous active executions: %s", data_size)
            if (data_size > 0):
                logger.info("Actual data update complete for all previous active executions which have actual data greater than (TP/3) days")
                for x in default_execs:
                    model_execution_id_val = x[0]
                    prediction_date = x[1]
                    logger.info("Impact Analysis started for model_execution_id: %s", model_execution_id_val)
                    logger.info("prediction_date: %s", prediction_date)
                    test = lia.actual_val_default_data_fetch(model_execution_id_val, brand_name, client_id, prediction_date, max_date, logger)
                    if test is not None:
                        lia.impact_analysis(int(model_execution_id_val), TP, prediction_date,
                                    max_date, conn_info, mysql_connection, analytics_query_config, test, logger)
                    else:
                        logger.info("Actual data not present for model_execution_id = %s", model_execution_id)
                        print("InComplete")
            else:
                logger.info("Actual data not present for previous active executions for this model")
                print("InComplete")
        else:
            logger.info("No previous active executions for this model")
            print("InComplete")
        logger.info("LTV impact analysis for previous active executions completed")
    except Exception as ex:
        logger.info(ex)

    print("API_Result=",Prediction_API.to_dict(orient='records'))
