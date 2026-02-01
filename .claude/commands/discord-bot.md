---
description: Work with the CS2 Analyzer Discord bot
---

# Discord Bot Development

Work with the CS2 Analyzer Discord bot.

## Bot Location

`src/opensight/discord_bot.py`

## Available Commands

| Command | Description |
|---------|-------------|
| `!analyze <url>` | Analyze a demo file from URL |
| `!decode <sharecode>` | Decode CS2 share code (CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx) |
| `!stats <steam_id>` | Get player statistics |
| `!coaching <steam_id>` | Get AI coaching advice |
| `!health` | Check API health status |
| `!maps` | List available maps |

## Environment Variables

Required:
```
DISCORD_BOT_TOKEN=your_bot_token_here
OPENSIGHT_API_URL=http://localhost:7860
```

## Run Locally

```cmd
set DISCORD_BOT_TOKEN=your_token
set OPENSIGHT_API_URL=http://localhost:7860
set PYTHONPATH=src
python -m opensight.discord_bot
```

## Adding a New Command

### Step 1: Define Command

```python
@bot.command(name="yourcommand")
async def your_command(ctx, arg1: str, arg2: Optional[int] = None):
    """Command description."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/your-endpoint") as response:
            if response.status == 200:
                data = await response.json()
                # Create embed
                embed = discord.Embed(
                    title="Your Title",
                    description="Description",
                    color=discord.Color.green()
                )
                embed.add_field(name="Field", value=data["value"])
                await ctx.send(embed=embed)
            else:
                await ctx.send("Error message")
```

### Step 2: Test Locally

1. Start the API: `uvicorn opensight.api:app --port 7860`
2. Start the bot: `python -m opensight.discord_bot`
3. Test in Discord

### Step 3: Error Handling

```python
@bot.command(name="yourcommand")
async def your_command(ctx, arg1: str):
    try:
        # Your logic
        pass
    except aiohttp.ClientError as e:
        await ctx.send(f"API connection error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in yourcommand: {e}")
        await ctx.send("An unexpected error occurred")
```

## Embed Best Practices

```python
embed = discord.Embed(
    title="Analysis Results",
    description=f"Demo: {demo_name}",
    color=discord.Color.blue(),
    timestamp=datetime.utcnow()
)

# Add fields
embed.add_field(name="HLTV Rating", value=f"{rating:.2f}", inline=True)
embed.add_field(name="K/D", value=f"{kd:.2f}", inline=True)

# Add footer
embed.set_footer(text="OpenSight CS2 Analyzer")

await ctx.send(embed=embed)
```

## File Uploads

For demo file uploads:
```python
@bot.command(name="analyze")
async def analyze(ctx):
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        if attachment.filename.endswith(('.dem', '.dem.gz')):
            # Process file
            file_bytes = await attachment.read()
            # Send to API
```

## Rate Limiting

The bot respects API rate limits:
- /analyze: 5/minute
- /replay/generate: 3/minute

Implement client-side waiting:
```python
import asyncio

async def rate_limited_request(session, url):
    async with rate_limit_lock:
        response = await session.get(url)
        if response.status == 429:
            await asyncio.sleep(60)  # Wait for rate limit reset
            return await rate_limited_request(session, url)
        return response
```

## Deployment

The bot runs separately from the main API. Deploy options:
1. Same server as API (different process)
2. Separate server pointing to API URL
3. Docker container

Remember to set environment variables in production.
