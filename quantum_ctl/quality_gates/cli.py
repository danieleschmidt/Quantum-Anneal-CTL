"""Quality Gates CLI Interface"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Optional, List
import click
import logging

from .gate_runner import QualityGateRunner
from .config import QualityGateConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Quantum-Anneal-CTL Quality Gates CLI"""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--gates', '-g', multiple=True, help='Specific gates to run')
@click.option('--output', '-o', type=click.Path(), help='Output directory for reports')
@click.option('--format', '-f', type=click.Choice(['json', 'html', 'both']), default='both', help='Report format')
@click.option('--fail-fast', is_flag=True, help='Stop on first failure')
@click.option('--parallel', is_flag=True, default=True, help='Run gates in parallel')
@click.pass_context
def run(ctx, config, gates, output, format, fail_fast, parallel):
    """Run quality gates"""
    
    # Load configuration
    if config:
        # TODO: Load from config file
        gate_config = QualityGateConfig()
    else:
        gate_config = QualityGateConfig.from_env()
    
    # Override config with CLI options
    if output:
        gate_config.report_output_dir = output
    if fail_fast:
        gate_config.fail_fast = fail_fast
    if not parallel:
        gate_config.parallel_execution = False
    
    gate_config.generate_html_report = format in ['html', 'both']
    
    # Run quality gates
    runner = QualityGateRunner(gate_config)
    
    async def run_gates():
        try:
            gate_filter = list(gates) if gates else None
            result = await runner.run_all_gates(gate_filter=gate_filter)
            
            # Exit with non-zero code if failed
            if not result.get('passed', False):
                sys.exit(1)
                
        except Exception as e:
            click.echo(f"❌ Quality gates failed with error: {e}", err=True)
            if ctx.obj.get('verbose'):
                import traceback
                traceback.print_exc()
            sys.exit(2)
    
    asyncio.run(run_gates())


@cli.command()
@click.argument('gate_name')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context  
def run_single(ctx, gate_name, config):
    """Run a single quality gate"""
    
    # Load configuration
    if config:
        gate_config = QualityGateConfig()
    else:
        gate_config = QualityGateConfig.from_env()
    
    runner = QualityGateRunner(gate_config)
    
    async def run_single_gate():
        try:
            result = await runner.run_single_gate(gate_name)
            
            # Print result
            click.echo(f"Gate: {result.gate_name}")
            click.echo(f"Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")
            click.echo(f"Score: {result.score:.1f}/100")
            click.echo(f"Execution Time: {result.execution_time_ms:.1f}ms")
            
            if result.messages:
                click.echo("Messages:")
                for msg in result.messages:
                    click.echo(f"  - {msg}")
            
            if not result.passed:
                sys.exit(1)
                
        except ValueError as e:
            click.echo(f"❌ Gate not found: {e}", err=True)
            sys.exit(2)
        except Exception as e:
            click.echo(f"❌ Gate execution failed: {e}", err=True)
            if ctx.obj.get('verbose'):
                import traceback
                traceback.print_exc()
            sys.exit(2)
    
    asyncio.run(run_single_gate())


@cli.command()
def list_gates():
    """List available quality gates"""
    config = QualityGateConfig()
    runner = QualityGateRunner(config)
    
    gates = runner.get_available_gates()
    
    click.echo("Available Quality Gates:")
    click.echo("=" * 50)
    
    for gate in gates:
        click.echo(f"  • {gate}")
    
    click.echo(f"\nTotal: {len(gates)} gates available")


@cli.command()
@click.option('--output', '-o', type=click.File('w'), default='-', help='Output file (default: stdout)')
def config_template(output):
    """Generate a configuration template"""
    
    config = QualityGateConfig()
    template = {
        "quality_gates": {
            "min_test_coverage": config.min_test_coverage,
            "max_api_response_time_ms": config.max_api_response_time_ms,
            "max_memory_usage_mb": config.max_memory_usage_mb,
            "security_scan_enabled": config.security_scan_enabled,
            "max_complexity": config.max_complexity,
            "min_docstring_coverage": config.min_docstring_coverage,
            "fail_fast": config.fail_fast,
            "parallel_execution": config.parallel_execution,
            "generate_html_report": config.generate_html_report,
            "report_output_dir": config.report_output_dir
        },
        "description": {
            "min_test_coverage": "Minimum test coverage percentage required (0-100)",
            "max_api_response_time_ms": "Maximum API response time in milliseconds", 
            "max_memory_usage_mb": "Maximum memory usage in MB",
            "security_scan_enabled": "Enable security vulnerability scanning",
            "max_complexity": "Maximum cyclomatic complexity for functions",
            "min_docstring_coverage": "Minimum documentation coverage percentage",
            "fail_fast": "Stop execution on first gate failure",
            "parallel_execution": "Run gates in parallel for better performance",
            "generate_html_report": "Generate HTML dashboard report",
            "report_output_dir": "Directory for output reports"
        }
    }
    
    json.dump(template, output, indent=2)
    
    if output.name != '<stdout>':
        click.echo(f"Configuration template saved to: {output.name}")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file to validate')
def validate_config(config):
    """Validate quality gates configuration"""
    
    if config:
        # TODO: Load and validate config file
        click.echo(f"Validating configuration file: {config}")
        gate_config = QualityGateConfig()
    else:
        click.echo("Validating environment-based configuration")
        gate_config = QualityGateConfig.from_env()
    
    runner = QualityGateRunner(gate_config)
    
    async def validate():
        try:
            validation = await runner.validate_configuration()
            
            if validation['valid']:
                click.echo("✅ Configuration is valid")
            else:
                click.echo("❌ Configuration has issues:")
                for issue in validation['issues']:
                    click.echo(f"  - {issue}")
                sys.exit(1)
                
            click.echo(f"\nConfigured gates: {len(validation['gates'])}")
            for gate in validation['gates']:
                status = "enabled" if gate['enabled'] else "disabled"
                click.echo(f"  • {gate['name']}: {status}")
                
        except Exception as e:
            click.echo(f"❌ Configuration validation failed: {e}", err=True)
            sys.exit(2)
    
    asyncio.run(validate())


@cli.command()
@click.argument('report_file', type=click.Path(exists=True))
def show_report(report_file):
    """Show summary of a quality gates report"""
    
    try:
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        # Extract key information
        overall_passed = report.get('passed', False)
        overall_score = report.get('score', 0)
        summary = report.get('summary', {})
        
        # Print summary
        status_icon = "✅" if overall_passed else "❌"
        status_text = "PASSED" if overall_passed else "FAILED"
        
        click.echo(f"\n{status_icon} Overall Status: {status_text}")
        click.echo(f"📊 Score: {overall_score:.1f}/100")
        click.echo(f"📈 Gates: {summary.get('passed_gates', 0)}/{summary.get('total_gates', 0)} passed")
        click.echo(f"⏱️  Time: {report.get('execution_time_ms', 0):.1f}ms")
        
        # Show gate details
        click.echo("\n📋 Gate Details:")
        for gate in report.get('gates', []):
            gate_icon = "✅" if gate.get('passed') else "❌"
            gate_name = gate.get('gate_name', 'Unknown').replace('_', ' ').title()
            gate_score = gate.get('score', 0)
            
            click.echo(f"  {gate_icon} {gate_name}: {gate_score:.1f}/100")
            
            # Show first message
            messages = gate.get('messages', [])
            if messages:
                click.echo(f"     └─ {messages[0]}")
        
    except json.JSONDecodeError:
        click.echo(f"❌ Invalid JSON report file: {report_file}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Failed to read report: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()