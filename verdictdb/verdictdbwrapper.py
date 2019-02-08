import pyverdict
import pandas as pd
import time


class Timer():

    def __init__(self):
        self.start_time = time.time()

    def get_seconds(self):
        return "%s" % (time.time() - self.start_time)


class logger():

    def __init__(self, logfile="verdict.log"):
        self.log = open(logfile, mode="a+")

    def __del__(self):
        self.log.close()

    def writeln(self, strs):
        self.log.write(str(strs))
        self.log.write("\n")
        print(strs)


def display(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


class VerdictWrapper():

    def __init__(self, host='localhost', logfile="verdict.log"):
        pyverdict.__version__              # prints pyverdict version
        pyverdict.__verdictdb_version__    # prints the core VerdictDB version
        self.verdict = pyverdict.mysql(host, 'hduser', 'password')
        self.log = logger(logfile)

    def create_scramble(self, table, size, ratio):
        """create a scramble from a given table

        Args:
            table (str): the table to be sampled from
            size (str): "10k", "100k", "1m"
            ratio (float): sample ratio
        """
        query = "create scramble verdict." + table + "_" + \
            str(size) + "_scramble" + " from verdict." + \
            table + " ratio " + "%12.8f"%(ratio)
        self.log.writeln("---------------------------------------------")
        self.log.writeln(query)
        timer = Timer()
        self.verdict.sql(query)
        self.log.writeln("Time cost is " + timer.get_seconds())
        self.log.writeln("---------------------------------------------")

    def create_CCPP_scrambles(self):
        """Create samples for the CCPP dataset.
        """
        self.create_scramble(table="powerplant", size="10k",  ratio=0.00000385)
        self.create_scramble(table="powerplant", size="100k", ratio=0.0000385)
        self.create_scramble(table="powerplant", size="1m",   ratio=0.000385)

    def create_tpcds_scrambles(self):
        pass

    def query(self, query):
        """execute a query from verdictdb.

        Args:
            query (String): the query

        Returns:
            list: [result from verdictdb, and the time cost]
        """
        self.log.writeln("---------------------------------------------")
        self.log.writeln(query)
        timer = Timer()
        result = self.verdict.sql(query)
        rs = result['c2'].tolist()[0]
        # self.log.writeln(result)
        # self.log.writeln(type(result))
        time_cost = timer.get_seconds()
        self.log.writeln("Prediction: " + str(rs))
        self.log.writeln("Time  cost: " + time_cost + " seconds.")
        self.log.writeln("---------------------------------------------")
        return rs, time_cost

    def query2file(self, file):
        results = []
        times = []
        self.log.writeln(
            "******************************************************************")
        self.log.writeln("Starting querying to " + file)
        self.log.writeln(
            "******************************************************************")
        with open(file, mode='r') as f:
            for line in f:
                rs, t = self.query(line)
                results.append(rs)
                times.append(t)
        self.log.writeln("Finished querying to file " + file)
        self.log.writeln(
            "******************************************************************")
        self.log.writeln(results)
        self.log.writeln(times)
        self.log.writeln(
            "******************************************************************")


def runCCPP():
    verdict = VerdictWrapper(host='137.205.118.65',
                             logfile="create_CCPP_scrambles.log")
    verdict = verdict.create_CCPP_scrambles()

    verdict = VerdictWrapper(host='137.205.118.65',
                             logfile="CCPP_small_query_range.log")
    verdict = verdict.query2file("../query/power/sql/verdict_query_sr_10k.sql")
    verdict = verdict.query2file(
        "../query/power/sql/verdict_query_sr_100k.sql")
    verdict = verdict.query2file("../query/power/sql/verdict_query_sr_1m.sql")

    verdict = VerdictWrapper(host='137.205.118.65',
                             logfile="CCPP_large_query_range.log")
    verdict = verdict.query2file("../query/power/sql/verdict_10k.sql")
    verdict = verdict.query2file("../query/power/sql/verdict_100k.sql")
    verdict = verdict.query2file("../query/power/sql/verdict_1m.sql")


if __name__ == "__main__":
    verdict = VerdictWrapper(host='137.205.118.65',logfile="create_CCPP_scrambles.log")
    verdict.verdict.sql("use verdict")
    rs=verdict.verdict.sql("show tables")
    display(rs)
    verdict.create_scramble(table="powerplant", size="1m",   ratio=0.000385)
    # verdict.create_scramble(table="powerplant", size="10k", ratio=3.85E-06)
    # verdict.create_CCPP_scrambles()
    # verdict.query("select count(*) from verdict.powerplant_10k_scramble")
    # verdict.query2file("../query/power/sql/verdict_query_sr_10k.sql")
    # runCCPP()
