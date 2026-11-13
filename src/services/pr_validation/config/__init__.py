"""
PR Validation Configuration Module.

This module provides configuration loading and management for the
PR validation system.

Configuration is loaded from validation_config.yaml which defines:
- Validation engine settings (timeouts, enabled phases)
- Risk calculator weights and thresholds
- Security, performance, and maintainability analysis settings
- GitHub integration configuration
- Feature flags

The configuration can be accessed via:
- Direct instantiation: ValidationConfig()
- Global singleton: get_validation_config()
- Dot notation access: config.get('validation_engine.overall_timeout')
"""
