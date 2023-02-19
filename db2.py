import jaydebeapi
import csv
import pandas as pd
import decimal
import jpype

# JDBC driver details
driver = 'com.ibm.db2.jcc.DB2Driver'
url = 'jdbc:db2://10.160.64.84:50000/hotmill'
user = 'ap'
password = 'baosight@1780'
jar_file = '/mnt/c/Users/wiki/AppData/Roaming/DBeaverData/drivers/maven/maven-central/com.ibm.db2/jcc-11.5.0.0.jar'
# Open database connection
conn = jaydebeapi.connect(driver, url, [user, password], jar_file)
# Execute the query
cursor = conn.cursor()

def search(query):
    # Query to execute
    cursor.execute(query)
    names = [desc[0] for desc in cursor.description]
    result_set = cursor.fetchall()
    # 将结果转换为Python的数据结构
    rows = []
    for row in result_set:
        new_row = []
        for col in row:
            # 将BigDecimal类型的值转换为Python的float类型
            if isinstance(col, jpype.JClass("java.math.BigDecimal")):
                new_row.append(float(decimal.Decimal(str(col))))
            else:
                new_row.append(col)
        rows.append(new_row)
    # 将10x20维的list转换为一个10行20列的DataFrame
    df_data = pd.DataFrame(rows, columns=names)

    return df_data

def TIME_GAP(strip_no,UnixTime):
    f = open("QueryBEAT.sql", "r")
    sql = f.read().replace("INPUT%%INPUT",str(int(strip_no)))
    df_data = search(sql)
    return  float(UnixTime - df_data.values)/60.0
def INDEX_ROLL(strip_no):
    f = open("INDEX_ROLL.sql", "r")
    sql = f.read().replace("INPUT%%INPUT",str(int(strip_no)))
    df_data = search(sql)
    return  float(df_data.values)
def DELTA_CLASS(strip_no):
    f = open("DELTA_CLASS.sql", "r")
    sql = f.read().replace("INPUT%%INPUT",str(int(strip_no)))
    df_data = search(sql)
    return   float(abs(df_data["THICK_CLASS"][0] - df_data["THICK_CLASS"][1]) + abs(df_data["TEMP_CLASS"][0] - df_data["TEMP_CLASS"][1]))
def LSAT_DELTA_TEMP(strip_no):
    f = open("LAST_DELTA_TEMP.sql", "r")
    sql = f.read().replace("INPUT%%INPUT",str(int(strip_no)))
    df_data = search(sql)
    return   float(df_data.values)

def query_one(strip_no):
    df = []
    f = open("QueryCalPost.sql","r")
    sql = f.read().replace("INPUT%%INPUT", str(int(strip_no)))
    df_data = search(sql)
    df_data = df_data.drop(labels=["ROLLGAP_OILROLL"], axis=1)
    # result_df = pd.concat([names, data], axis=1)
    # 将10维的list作为列名
    # df_data = df_data.transpose()

    row = []
    namesN = []
    names6 = ['STRIP_NO','STAND_NO', 'CORR_FORCE_STAND', 'DETAL_FORCE_CAL', 'DETAL_FORCE_POST',
              'KM', 'TEMP_DELTA', 'TEMP_CORR', 'GAP_DELTA', 'CORR_ZEROPOINT_USE', 'MILLSTRETCH_ROLL',
              'DELTA_MILL', 'ENTRY_TENSION']
    names7 = ['STAND_NO', 'FORCE_ACT', 'CORR_FORCE_STAND', 'CORR_FORCE_DELTA', 'DETAL_FORCE_CAL',
              'DETAL_FORCE_POST', 'DELTA_SPEED', 'DELTA_REDU', 'STRIP_WIDTH', 'ROLL_DIAM', 'KM',
              'CHEM_COEFF', 'FM_TEMP', 'TEMP_DELTA', 'TEMP_CORR', 'RM_TEMP_DELTA', 'FET_ACT_TEMP',
              'DESC_SUM', 'GAP_DELTA', 'CORR_ZEROPOINT_USE', 'DELTA_ZEROPOINT', 'MILLSTRETCH_ROLL',
              'DELTA_MILL', 'ENTRY_THICK', 'ROLLWEAR', 'ENTRY_TENSION',
              'BEND_FORCE','DELTA_THICK']
    names6compile = [name + "_6" for name in names6]
    names7compile = [name + "_7" for name in names7]
    otherComplie = ["WATER_FLOW","DELTA_WATER","BEAT","INDEX_ROLL","DELTA_CLASS","LSAT_DELTA_TEMP"]
    # 创建一个空的 DataFrame  .append(names7compile)
    names6compile += names7compile
    names6compile += otherComplie

    dfComplie = pd.DataFrame(columns=names6compile)

    water_flow = 0
    water_delta = 0
    for row in df_data.iterrows():
        row = row[1]
        if row["STAND_NO"] == 6.0:
            df += list(row[names6])
            water_flow += row["WATER_FLOW"]
            water_delta += row["DELTA_WATER"]

        elif row["STAND_NO"] == 7.0:

            water_flow += row["WATER_FLOW"]
            water_delta += row["DELTA_WATER"]
            # OneValue(int(row["STRIP_NO"]),row["UNIX_TIME"])
            # INDEX_ROLL(int(row["STRIP_NO"]))
            # DELTA_CLASS(int(row["STRIP_NO"]))
            # LSAT_DELTA_TEMP(int(row["STRIP_NO"]))
            df += list(row[names7])
            equipment = [water_flow,water_delta,TIME_GAP(int(row["STRIP_NO"]),row["UNIX_TIME"]),
                         INDEX_ROLL(int(row["STRIP_NO"])),DELTA_CLASS(int(row["STRIP_NO"])),
                         LSAT_DELTA_TEMP(int(row["STRIP_NO"]))]
            df += equipment

            indexsize = dfComplie.index.size
            dfComplie.loc[indexsize] = df
            df = []
            water_flow = 0
            water_delta = 0
        else:
            water_flow += row["WATER_FLOW"]
            water_delta += row["DELTA_WATER"]
    print("ok")

    return dfComplie





strip_no = '220206104400'
df = query_one(strip_no)
# Write the results to a CSV file
df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)


df.to_csv("output.csv")

# Close the database connection
conn.close()

