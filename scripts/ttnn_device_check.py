try:
    import ttnn
    print("ttnn imported OK")
    try:
        dev = ttnn.open_device(0)
        print("device opened:", dev)
        ttnn.close_device(dev)
    except Exception as e:
        print("ttnn present but device open failed (expected without VM):", e)
except Exception as e:
    print("ttnn not installed:", e)
