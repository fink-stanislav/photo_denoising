/**
 * Reads image from file input
 */
function readURL(input) {
	if (input.files && input.files[0]) {
		var reader = new FileReader();

		reader.onload = function(e) {
			$('#image')
				.attr('src', e.target.result)
				.width(224)
				.height(224);
		};

		reader.readAsDataURL(input.files[0]);
	}
}

$(document).ready(function() {

	$("#add_noise").click(function() {
		$("#add_noise").addClass("is-loading")
		$("#add_noise_form").submit()
	})

	$("#add_noise_form").submit(function(e) {
	    e.preventDefault();
	    var formData = new FormData(this);    

	    $.ajax({
	        url: $(this).attr("action"),
	        type: 'POST',
	        data: formData,
	        success: function (data) {
	            $("#add_noise").removeClass("is-loading")
	            $("#noisy_image").attr("src", 'get_image/' + data['noisy_image'])
	            $("#noisy_psnr").html(data['psnr'])
	        },
	        error: function (data) {
	            $("#add_noise").removeClass("is-loading")
	            $("#noisy_image").attr("src", "../static/placeholder_224.png")
	        },
	        cache: false,
	        contentType: false,
	        processData: false
	    });
	});

	$("#remove_noise").click(function() {
		$("#remove_noise").addClass("is-loading")
		$("#remove_noise_form").submit()
	})

	$("#remove_noise_form").submit(function(e) {
	    e.preventDefault();
	    var formData = new FormData(this);    

	    $.ajax({
	        url: $(this).attr("action"),
	        type: 'POST',
	        data: formData,
	        success: function (data) {
	            $("#remove_noise").removeClass("is-loading")
	            $("#download").attr("href", 'download/' + data['denoised_image'])
	            $("#denoised_image").attr("src", 'get_image/' + data['denoised_image'])
	            $("#denoised_psnr").html(data['psnr'])
	        },
	        error: function (data) {
	            $("#remove_noise").removeClass("is-loading")
	        },
	        cache: false,
	        contentType: false,
	        processData: false
	    });
	});

	$("#intencity").on('input', function () {
		$("#intencity_value").html(this.value + ' %')
	})

	$("#min_loss").on('input', function () {
		$("#min_loss_value").html(this.value)
	})

	$("#steps").on('input', function () {
		$("#steps_value").html(this.value)
	})

})
