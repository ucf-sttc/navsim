#!/bin/sh
# Auto-generated install.sh script for starting/helping the test profile creation process...


echo "#!/bin/sh
# the actual running/execution of the test, etc... This is called at run-time.
# The program under test and/or any parsing/wrapper scripts should then pipe the results to \$LOG_FILE for parsing.
# Passed to the script as arguments are any of the test arguments/options as defined by the test-definition.xml.

# Editing the test profile's results-definition.xml controls how the Phoronix Test Suite will capture the program's result.
# STATIC EXAMPLE below coordinated with the stock result-definition.xml.
echo \"Result: 55.5\" > \$LOG_FILE
echo \$? > ~/test-exit-status

" > ~/navsim-test-1
chmod +x ~/navsim-test-1

# Check out the `phoronix-test-suite debug-run` command when trying to debug your install/run behavior
