# AWS S3

## Permissions

Example: allow access to a remote at 35.234.11.198, or even local access (171.18.50.137).

```
{
    "Version": "2012-10-17",
    "Id": "S3PolicyId1",
    "Statement": [
        {
            "Sid": "IPAllow",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": "arn:aws:s3:::deoldify-images-repo/*",
            "Condition": {
                "NotIpAddress": {
                    "aws:SourceIp": [
                        "35.234.11.198",
                        "171.18.50.137"
                    ]
                }
            }
        }
    ]
}
```
