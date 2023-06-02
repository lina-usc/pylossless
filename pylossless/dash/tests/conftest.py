# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
# License: MIT

"""Configure the dash tests with the chrom webdrive for CI."""


from selenium.webdriver.chrome.options import Options


# As per https://community.plotly.com/t/dash-integration-testing-
# with-selenium-and-github-actions/43602/2
def pytest_setup_options():
    """Configure the dash tests with the chrom webdrive for CI."""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    return options
