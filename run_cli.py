#!/usr/bin/env python3
"""
Quick launcher for ONIKS user-friendly CLI.
Run this to test the new interface.
"""

if __name__ == "__main__":
    from oniks.ui.cli import UserFriendlyCLI
    
    cli = UserFriendlyCLI()
    cli.run()