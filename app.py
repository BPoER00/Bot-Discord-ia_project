import discord
from discord.ext import commands
import datetime
botToken = "[TOKEN DISCORD BOT]"

bot = commands.Bot(command_prefix=">", intents=discord.Intents.all(), description="Bot del curso IA, UMG\n Alumnos\n Bryan Paz.\n Emmanuel Cabrebra.")

@bot.command()
async def ping(ctx):
    await ctx.send("pong")

@bot.command()
async def sum(ctx, num1: int, num2: int):
    await ctx.send(f"Suma: {num1 + num2}")

@bot.command()
async def info(ctx):
    embedform = discord.Embed(
        title=f"{ctx.guild.name}",
        description="Opciones para utilizar",
        color=discord.Color.purple()
        )
    
    embedform.set_thumbnail(url="https://us.123rf.com/450wm/dxinerz/dxinerz1507/dxinerz150701509/42482050-ajustes-controles-de-imagen-de-icono-de-vector-opciones-tambi%C3%A9n-se-puede-utilizar-para-el-tel%C3%A9fono.jpg")

    await ctx.send(embed=embedform)
    
@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Streaming(name="Ia Project", url="github.com/BPoER00"))
    print(f"Hola, soy {bot.user}")

bot.run(botToken)
