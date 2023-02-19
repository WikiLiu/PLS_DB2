import jaydebeapi


class Db2Connect(object):

    def __init__(self):
        self.driver = 'com.ibm.PLS_DB2.jcc.DB2Driver'
        self.url = 'jdbc:PLS_DB2://10.160.64.81:50000/hotmill'
        self.user = 'ap'
        self.password = 'baosight@1780'
        self.jar_file = '/mnt/c/Users/wiki/AppData/Roaming/DBeaverData/drivers/maven/maven-central/com.ibm.PLS_DB2/jcc-11.5.0.0.jar'
        self.conn = jaydebeapi.connect(self.driver, self.url, [self.user, self.password], self.jar_file)

    def exec(self, query):
        self.cursor = self.conn.cursor()
        self.cursor.execute(query)
        names = [desc[0] for desc in self.cursor.description]
        rows =self.cursor.fetchall()
        self.cursor.close()

        return names,rows

    def __del__(self):
        self.conn.close()
