from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
from langchain.llms import Ollama