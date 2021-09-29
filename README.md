# This is the passenger_forecasting package.

passenger_forecasting is a complete package that makes passenger volume
predictions (that is; how many people will fly on a given flight) using
a variety of different models. SQL data pulls, data cleaning and preparation,
as well as model testing metrics are all included.

*****************************************************************************************
# NOTE
This package was produced by myself for a commercial client. Since their data is
commercial property, the majority of the data in the '/data' directory - some of which
is crucial to the operation of this package - has been removed. The pdf documentation also
makes frequent reference to the data in this directory, so be aware.

Furthermore, the 'populate_sandboxes.py' script relies on a non-open-source module (cadspy),
so this particular script will not work. This script is not crucial to the operation of the
package, and has been left in for reference.
*****************************************************************************************

All the information you need to use the code is in 'passenger_forecasting_code_guide.pdf' .

'structure_of_input_vectors.pdf' describes how the input vectors are constructed,
and why they have varying length depending upon how far out you wish to forecast,
and how frequently you sample the snapshots.

Jack Whaley-Baldwin

September 2021

