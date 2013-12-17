#!/usr/bin/env python
import MySQLdb as mdb

"""
DB_VARS = (
  'mysql.csail.mit.edu',
  'ajratner',
  'ajpassword',
  'zkh0330'
  )
"""

DB_VARS = (
  'localhost',
  'statutes',
  'rgb',
  'statutes'
  )


# simple mysql connection class to be used in "with" clause
# returns a 'handle' on the database = (conn, cur)
class DB_connection:
  def __init__(self, db_vars=DB_VARS, db_type='MYSQL'):

    # db_vars = (DB_HOST, DB_USER, DB_PWD, DB_NAME)
    self.db_vars = db_vars
    self.db_type = db_type

  
  # open connection & cursor
  def __enter__(self):
    self.conn = mdb.connect(*self.db_vars)
    self.cur = self.conn.cursor()
    return (self.conn, self.cur)

  
  # close connection & cursor
  def __exit__(self, *args):
    self.cur.close()
    self.conn.close()

