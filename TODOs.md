
document the changes in the changelog for the new planned version 0.0.3
include the design decisions and rationales behind them (found in AGGRESSIVE_WORKAROUND_PLAN.md)


MATURITY


clean up all outdated prototype and experimentation code and results.
only keep usage examples useful for the users or developers.



add a demo script for comparing different seaborn styles side by side:
styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]
for style in styles:
    with sns.axes_style(style):



refactor this repository: the project's official name now is "seaborn-paracoords"
Clean up all other names and make everything consistent.


get rid of all local imports and move them to the top of the module





DOCUMENTATION


include remarks in the project documentation that this project aims at wrapping and extending seaborn code as much as possible without duplicating functionality.
this information should be obvious for coding agents and human contributors alike.
also highlight the contributions of this package: This library fills a genuine gap in the Python ecosystem. The independent axis scaling feature alone makes it irreplaceable for real-world multi-variable data exploration.



document clearly to the user that the ordering of the columns in the dataframe will be maintained for plotting.


Mention the design decision to keep the core logic inside a single module file. This file can be downloaded and used standalone if needed.


Add the instruction to alway run the pre-commit hooks after every completed change with "make hook" to the Agents.md file


EDGE CASE HANDLING
desired edge case handling: as soon as there are entries of None, inf etc. treat the column as a non-continous categorical column.
output a warning in such cases.

skip plotting categorical columns with more than 100 distinct values. output a warning in such cases.

document this behavior. and let the user know they must handle such cases themselves. e.g. missing value imputation.
