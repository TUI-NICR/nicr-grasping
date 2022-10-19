import pytest

@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.benchmark(group="io")
def test_rectangle_grasp_list_save_benchmark(benchmark, rectangle_grasp_list, tmp_path, compress):
    grasp_file = tmp_path / 'graspfile.pkl'
    benchmark(rectangle_grasp_list.save, grasp_file, compressed=compress)
