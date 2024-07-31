from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import bcrypt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedeltapip
import bz2
import pickle