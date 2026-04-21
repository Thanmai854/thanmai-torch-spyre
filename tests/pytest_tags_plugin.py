"""
Pytest plugin to capture test tags and add them to XML reports.

This plugin hooks into pytest's test collection and reporting phases to:
1. Extract tags from pytest marks applied to test methods
2. Add tag information to the XML report as properties

Usage:
    Add to conftest.py:
        pytest_plugins = ["pytest_tags_plugin"]
    
    Or use via command line:
        pytest --conftest=pytest_tags_plugin.py
"""

import pytest
from typing import Dict, List


class TagsPlugin:
    """Plugin to capture and report test tags in XML output."""
    
    def __init__(self):
        self.test_tags: Dict[str, List[str]] = {}
    
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        """Hook to capture test tags and add them to the report."""
        outcome = yield
        report = outcome.get_result()
        
        # Only process on setup phase to avoid duplicates
        if report.when == "setup":
            # Extract tags from pytest marks
            tags = []
            for mark in item.iter_markers():
                # Skip built-in pytest marks
                if mark.name not in ['parametrize', 'skip', 'skipif', 'xfail', 
                                      'usefixtures', 'filterwarnings', 'timeout']:
                    tags.append(mark.name)
            
            if tags:
                # Store tags for this test
                self.test_tags[item.nodeid] = tags
                
                # Add tags as user properties for XML report
                if not hasattr(report, 'user_properties'):
                    report.user_properties = []
                report.user_properties.append(('tags', ','.join(sorted(tags))))
    
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_logreport(self, report):
        """Ensure tags are preserved in all report phases."""
        outcome = yield
        
        # Add tags to call and teardown phases if they exist
        if report.when in ['call', 'teardown'] and report.nodeid in self.test_tags:
            tags = self.test_tags[report.nodeid]
            if not hasattr(report, 'user_properties'):
                report.user_properties = []
            # Check if tags already added to avoid duplicates
            if not any(key == 'tags' for key, _ in report.user_properties):
                report.user_properties.append(('tags', ','.join(sorted(tags))))


def pytest_configure(config):
    """Register the plugin with pytest."""
    config.pluginmanager.register(TagsPlugin(), "tags_plugin")

# Made with Bob
