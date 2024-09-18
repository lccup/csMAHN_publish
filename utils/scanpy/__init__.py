#!/usr/bin/env python
# coding: utf-8

# ```mermaid
# graph LR;
#     __init__{{__init__}};
#     general[general];
#     general[general] -.-> |*| __init__;
# 
#     subgraph scanpy
#         scanpy.__init__{{__init__}};
#         scanpy.sc[sc];
#         scanpy.pl[pl];
#         scanpy.sc -->scanpy.pl
#         scanpy.pl -.-> scanpy.__init__
#         scanpy.sc -.-> |*| scanpy.__init__
#     end
#     general --> scanpy.sc
#     scanpy.__init__ -.-> __init__
# 
# ```

# In[ ]:


from utils.scanpy.sc import *
from utils.scanpy import pl

