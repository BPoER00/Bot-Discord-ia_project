import discord
from discord.ext import commands
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

botToken = "[TOKEN]"

bot = commands.Bot(command_prefix=">", intents=discord.Intents.all(), description="Bot del curso IA, UMG\n Alumnos\n Bryan Paz.\n Emmanuel Cabrebra.")

#Operaciones
#Suma
@bot.command()
async def sum(ctx, num1: int, num2: int):

    xt = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    yt = np.array([[0.0], [1.0], [1.0], [2.0]])

    oculta1 = tf.keras.layers.Dense(units=3, input_shape=(2,))
    oculta2 = tf.keras.layers.Dense(units=3)
    salida = tf.keras.layers.Dense(units=1)
    model = tf.keras.Sequential([oculta1, oculta2, salida])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    await ctx.send("La red neuronal esta procesando...")
    historial = model.fit(xt, yt, epochs=1000, batch_size=4)

    # plt.xlabel("# epoca")
    # plt.ylabel("Magnitud de perdida")
    # plt.plot(historial.history['loss'])
    # plt.show()

    xtest = np.array([[num1, num2]])

    prediccion = model.predict(xtest)
    await ctx.send("Resultado de la suma es: " + str(prediccion))

#Resta
@bot.command()
async def res(ctx, num1: int, num2: int):

    xt = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    yt = np.array([[0.0], [-1.0], [1.0], [0.0]])

    oculta1 = tf.keras.layers.Dense(units=32, input_shape=(2,))
    oculta2 = tf.keras.layers.Dense(units=32)
    salida = tf.keras.layers.Dense(units=1)
    model = tf.keras.Sequential([oculta1, oculta2, salida])

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    #     tf.keras.layers.Dense(1)
    # ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    await ctx.send("La red neuronal esta procesando...")
    historial = model.fit(xt, yt, epochs=1000, batch_size=4)

    plt.xlabel("# epoca")
    plt.ylabel("Magnitud de perdida")
    plt.plot(historial.history['loss'])
    plt.show()

    xtest = np.array([[num1, num2]])

    prediccion = model.predict(xtest)
    await ctx.send("Resultado de la resta es: " + str(prediccion))

@bot.command()
async def mul(ctx, num1: int, num2: int):

    xt = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 20.0],
                   [1.0, 1.0], [1.0, 3.0], [6.0, 1.0], [17.0, 1.0],
                   [2.0, 2.0], [2.0, 4.0], [8.0, 2.0], [15.0, 2.0],
                   [3.0, 3.0], [3.0, 5.0], [7.0, 3.0], [12.0, 3.0],
                   [4.0, 4.0], [4.0, 2.0], [5.0, 4.0], [22.0, 4.0],
                   [5.0, 5.0], [5.0, 6.0], [9.0, 5.0], [19.0, 5.0],
                   [6.0, 6.0], [6.0, 7.0], [4.0, 6.0], [10.0, 6.0],
                   [7.0, 7.0], [7.0, 2.0], [3.0, 7.0], [13.0, 7.0],
                   [8.0, 8.0], [8.0, 9.0], [2.0, 8.0], [30.0, 8.0],
                   [9.0, 9.0], [9.0, 8.0], [0.0, 9.0], [16.0, 9.0]])
    
    yt = np.array([[0.0], [0.0], [0.0], [0.0],
                   [1.0], [3.0], [6.0], [17.0],
                   [4.0], [8.0], [16.0], [30.0],
                   [9.0], [15.0], [21.0], [36.0],
                   [14.0], [8.0], [20.0], [88.0],
                   [25.0], [30.0], [45.0], [95.0],
                   [36.0], [42.0], [24.0], [60.0],
                   [49.0], [14.0], [21.0], [91.0],
                   [64.0], [72.0], [16.0], [240.0],
                   [81.0], [72.0], [0.0], [144.0]])

    oculta1 = tf.keras.layers.Dense(units=100, input_shape=(2,), activation='relu')
    oculta2 = tf.keras.layers.Dense(units=100, activation='relu')
    salida = tf.keras.layers.Dense(units=1, activation='linear')
    model = tf.keras.Sequential([oculta1, oculta2, salida])
    
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, input_shape=(2,), activation='relu'),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='linear')
    # ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    await ctx.send("La red neuronal esta procesando...")
    historial = model.fit(xt, yt, epochs=1000, batch_size=4)

    plt.xlabel("# epoca")
    plt.ylabel("Magnitud de perdida")
    plt.plot(historial.history['loss'])
    plt.show()

    xtest = np.array([[num1, num2]])

    prediccion = model.predict(xtest)
    await ctx.send("Resultado de la multiplicacion es: " + str(prediccion))

@bot.command()
async def div(ctx, num1: int, num2: int):

    xt = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 20.0],
                   [1.0, 1.0], [1.0, 3.0], [6.0, 1.0], [17.0, 1.0],
                   [2.0, 2.0], [2.0, 4.0], [8.0, 2.0], [15.0, 2.0],
                   [3.0, 3.0], [3.0, 5.0], [7.0, 3.0], [12.0, 3.0],
                   [4.0, 4.0], [4.0, 2.0], [5.0, 4.0], [22.0, 4.0],
                   [5.0, 5.0], [5.0, 6.0], [9.0, 5.0], [19.0, 5.0],
                   [6.0, 6.0], [6.0, 7.0], [4.0, 6.0], [10.0, 6.0],
                   [7.0, 7.0], [7.0, 2.0], [3.0, 7.0], [13.0, 7.0],
                   [8.0, 8.0], [8.0, 9.0], [2.0, 8.0], [30.0, 8.0],
                   [9.0, 9.0], [9.0, 8.0], [0.0, 9.0], [16.0, 9.0]])
    
    yt = np.array([[0.0], [0.0], [0.0], [0.0],
                   [1.0], [0.3], [6.0], [17.0],
                   [1.0], [0.5], [4.0], [7.5],
                   [1.0], [0.6], [2.3], [4.0],
                   [1.0], [2.0], [1.25], [5.5],
                   [1.0], [0.5], [1.8], [3.8],
                   [1.0], [0.85], [0.67], [1.67],
                   [1.0], [3.5], [0.43], [1.85],
                   [1.0], [0.89], [0.25], [3.75],
                   [1.0], [1.125], [0.0], [1.77]])

    oculta1 = tf.keras.layers.Dense(units=100, input_shape=(2,), activation='relu')
    oculta2 = tf.keras.layers.Dense(units=100, activation='relu')
    salida = tf.keras.layers.Dense(units=1, activation='linear')
    model = tf.keras.Sequential([oculta1, oculta2, salida])
    
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, input_shape=(2,), activation='relu'),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='linear')
    # ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    await ctx.send("La red neuronal esta procesando...")
    historial = model.fit(xt, yt, epochs=1000, batch_size=4)

    plt.xlabel("# epoca")
    plt.ylabel("Magnitud de perdida")
    plt.plot(historial.history['loss'])
    plt.show()

    xtest = np.array([[num1, num2]])

    prediccion = model.predict(xtest)
    await ctx.send("Resultado de la division es: " + str(prediccion))

@bot.command()
async def CaF(ctx, n1: int):
    celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
    fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

    oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
    oculta2 = tf.keras.layers.Dense(units=3)
    salida = tf.keras.layers.Dense(units=1)
    model = tf.keras.Sequential([oculta1, oculta2, salida])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )

    await ctx.send("La red neuronal esta procesando...")
    historial = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)

    plt.xlabel("# epoca")
    plt.ylabel("Magnitud de perdida")
    plt.plot(historial.history['loss'])
    plt.show()

    xtest = np.array([[n1]])
    prediccion = model.predict(xtest)

    await ctx.send("Resultado de la conversion de Grado Celsius a Grado Fahrenheit es: " + str(prediccion))

@bot.command()
async def FaC(ctx, n1: int):
    celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
    fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

    oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
    oculta2 = tf.keras.layers.Dense(units=3)
    salida = tf.keras.layers.Dense(units=1)
    model = tf.keras.Sequential([oculta1, oculta2, salida])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )

    await ctx.send("La red neuronal esta procesando...")
    historial = model.fit(fahrenheit, celsius, epochs=1000, verbose=False)

    plt.xlabel("# epoca")
    plt.ylabel("Magnitud de perdida")
    plt.plot(historial.history['loss'])
    plt.show()

    xtest = np.array([[n1]])
    prediccion = model.predict(xtest)

    await ctx.send("Resultado de la conversion de Grado Fahrenheit a Grado Celsius es: " + str(prediccion))

@bot.command()
async def info(ctx):
    embedform = discord.Embed(
        title=f"{ctx.guild.name}",
        description="Opciones para utilizar",
        color=discord.Color.purple()
        )
    embedform.add_field(name="Comandos de operaciones aritmeticas con redes neuronales ", value=">sum 1 1 (Realiza una Suma)\n >res 1 1 (Realiza una resta)\n >mul 1 1 (Realiza una multi)\n >div 1 1 (Realiza una division)\n")
    embedform.add_field(name="Comandos de operaciones convertidor con redes neuronales", value=">CaF 1 (Convierte de C a F)\n >FaC 1 (Convierte de F a C)\n") 
    embedform.set_thumbnail(url="https://us.123rf.com/450wm/dxinerz/dxinerz1507/dxinerz150701509/42482050-ajustes-controles-de-imagen-de-icono-de-vector-opciones-tambi%C3%A9n-se-puede-utilizar-para-el-tel%C3%A9fono.jpg")

    await ctx.send(embed=embedform)
    
@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Streaming(name="Ia Project", url="github.com/BPoER00"))
    print(f"Hola, soy {bot.user}")

bot.run(botToken)
