# TODOs



## FEATURE


input parameter for adding an additional label, e.g. for describing the shared units of numeric axes (when all axes share the same unit). for horizontal orientation this would be a label below the x-axis, for vertical orientation a label left of the y-axis. the style should match the axis tick labels, the font should be slightly smaller than the tick labels.

input parameter for tick label formatting. could be a format string or a callable function. this should be applied to all axes, both categorical and numeric.



### BUGFIXes

the "grid" styles are not properly applied. the grid lines are missing

related: make ticks align with grid lines
see aggressive_workaround_demo.py for a demonstration of working code. this is however not using seaborn utils as desired.



## MATURITY

simplify the test cases. use parameterisation wherever possible. apply the DRY principle


clean up all outdated prototype and experimentation code and results.
only keep usage examples useful for the users or developers.


refactor this repository: the project's official name now is "seaborn-paracoords"
Clean up all other names and make everything consistent.


get rid of all local imports and move them to the top of the module





## DOCUMENTATION


include remarks in the project documentation that this project aims at wrapping and extending seaborn code as much as possible without duplicating functionality.
this information should be obvious for coding agents and human contributors alike.
also highlight the contributions of this package: This library fills a genuine gap in the Python ecosystem. The independent axis scaling feature alone makes it irreplaceable for real-world multi-variable data exploration.



document clearly to the user that the ordering of the columns in the dataframe will be maintained for plotting.


Mention the design decision to keep the core logic inside a single module file. This file can be downloaded and used standalone if needed.



## EDGE CASE HANDLING

desired edge case handling: as soon as there are entries of None, inf etc. treat the column as a non-continous categorical column.
output a warning in such cases.

skip plotting categorical columns with more than 100 distinct values. output a warning in such cases.

document this behavior. and let the user know they must handle such cases themselves. e.g. missing value imputation.
