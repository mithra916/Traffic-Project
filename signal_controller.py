def allocate_green_time(vehicle_count):
    if vehicle_count > 15:
        return 80
    elif vehicle_count > 5:
        return 40
    else:
        return 20
