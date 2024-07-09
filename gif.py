import imageio


def video_to_gif(input_path, output_path, fps=10):
    reader = imageio.get_reader(input_path)
    writer = imageio.get_writer(output_path, fps=fps)

    for frame in reader:
        writer.append_data(frame)

    writer.close()

if __name__ == "__main__":
    video_to_gif('outputs/Med_Djo_cut_tracked.avi', 'outputs/Med_Djo_cut_tracked.gif', fps=24)