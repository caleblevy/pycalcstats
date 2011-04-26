#!/usr/bin/env python2

# Build test data for later use (depends on Python and numpy).

import numpy as np
import pickle
import zipfile

#
# Generate random data.
#
data = np.random.rand(10000)
results = {}
a, b = data.min(), data.max()
# Need to convert results to regular floats in case numpy isn't available
# when the data is read back in.
results['midrange'] = float((a+b)/2)
results['range'] = float(b-a)
results['mean'] = float(data.mean())
results['pstdev'] = float(data.std())  # population standard deviation
results['pvariance'] = float(data.var())  # population variance
results['sum'] = float(data.sum())
results['product'] = float(data.prod())

#
# Store data in a zipped pickle file.
#
zf = zipfile.ZipFile('test_data.zip', 'w', zipfile.ZIP_DEFLATED)
zf.writestr('data.pkl', pickle.dumps(list(map(float, data)), -1))
zf.writestr('results.pkl', pickle.dumps(results, -1))
zf.close()

