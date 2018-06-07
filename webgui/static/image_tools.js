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
	        },
	        error: function (data) {
	            $("#remove_noise").removeClass("is-loading")
	        },
	        cache: false,
	        contentType: false,
	        processData: false
	    });
	});
	
})
