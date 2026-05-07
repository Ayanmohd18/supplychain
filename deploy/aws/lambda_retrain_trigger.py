"""
AWS Lambda function that:
1. Checks CloudWatch metrics for WMAPE degradation
2. Triggers EC2 retraining job if WMAPE > threshold
3. Sends SNS notification
"""

import os
import boto3
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger()
logger.setLevel(logging.INFO)

cloudwatch = boto3.client('cloudwatch')
ssm = boto3.client('ssm')
sns = boto3.client('sns')

# Environment variables
WMAPE_THRESHOLD = float(os.getenv('WMAPE_THRESHOLD', '0.60'))
EC2_INSTANCE_ID = os.getenv('EC2_INSTANCE_ID', 'i-0123456789abcdef0') # Retraining instance
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN', 'arn:aws:sns:us-east-1:123456789012:RetrainAlerts')

def get_latest_wmape():
    """Fetches the latest WMAPE metric from CloudWatch."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)
    
    response = cloudwatch.get_metric_statistics(
        Namespace='SupplyChain/Forecasting',
        MetricName='WMAPE',
        Dimensions=[{'Name': 'Model', 'Value': 'TFT'}],
        StartTime=start_time,
        EndTime=end_time,
        Period=86400,
        Statistics=['Average']
    )
    
    datapoints = response.get('Datapoints', [])
    if not datapoints:
        return None
        
    # Get the most recent datapoint
    latest_datapoint = sorted(datapoints, key=lambda x: x['Timestamp'])[-1]
    return latest_datapoint['Average']

def trigger_retraining():
    """Triggers an SSM Run Command on the EC2 training instance."""
    logger.info(f"Triggering retraining on EC2 instance {EC2_INSTANCE_ID}...")
    
    commands = [
        "cd /opt/supplychain",
        "source venv/bin/activate",
        "python scripts/train_tft.py",
        "python deploy/aws/s3_upload.py"
    ]
    
    response = ssm.send_command(
        InstanceIds=[EC2_INSTANCE_ID],
        DocumentName="AWS-RunShellScript",
        Parameters={'commands': commands}
    )
    return response['Command']['CommandId']

def send_sns_alert(wmape, command_id):
    """Sends an SNS notification about the retraining event."""
    message = (
        f"🚨 Model Performance Degradation Alert 🚨\\n\\n"
        f"The TFT forecasting model's WMAPE has reached {wmape:.4f}, "
        f"which exceeds the threshold of {WMAPE_THRESHOLD}.\\n\\n"
        f"Automated retraining has been triggered on EC2 instance {EC2_INSTANCE_ID}.\\n"
        f"SSM Command ID: {command_id}\\n"
    )
    
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="[Alert] M5 Forecast Retraining Triggered",
        Message=message
    )

def lambda_handler(event, context):
    try:
        current_wmape = get_latest_wmape()
        
        if current_wmape is None:
            logger.info("No WMAPE metrics found in CloudWatch for the past 24 hours.")
            return {'statusCode': 200, 'body': 'No metrics found.'}
            
        logger.info(f"Current WMAPE: {current_wmape:.4f} (Threshold: {WMAPE_THRESHOLD})")
        
        if current_wmape > WMAPE_THRESHOLD:
            logger.warning("WMAPE threshold exceeded! Initiating retraining sequence.")
            command_id = trigger_retraining()
            send_sns_alert(current_wmape, command_id)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Retraining triggered successfully',
                    'wmape': current_wmape,
                    'command_id': command_id
                })
            }
        else:
            logger.info("WMAPE is within acceptable limits. No action taken.")
            return {
                'statusCode': 200,
                'body': 'Performance is optimal.'
            }
            
    except Exception as e:
        logger.error(f"Error checking metrics or triggering retraining: {str(e)}")
        raise e
