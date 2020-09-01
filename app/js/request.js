//$base_url = 'http://10.168.50.173/age-gender/'
$base_url = 'http://127.0.0.1:8050/'
//$base_url = 'https://uat-facenet.pnbmetlife.com/age-gender/'

$loading = $('#loading-div')
$recognize_modal = $('#recognize-modal')
$modal = $('#modal')

$data = {}

$(document).ajaxStart(function() {
  $loading.show();
});

$(document).ajaxComplete(function() {
  $loading.hide();
});

$(document).ajaxError(function() {
  add_modal_text_and_display($modal, "Some Error Occured, Please Try Again!")
});

//make a post request to recognize API
function predict(b64){
    $.ajax({
      type: "POST",
      url: $base_url + "age_gender_pred",
      data: {image : b64},
      dataType: "json",
      success: function(result) {
        console.log(result);
        if(result){
          $data = result;
          if(parseFloat(result.gender.prob) > 0.8){
            add_modal_text_and_display($modal, "Gender: " + result.gender.pred + "<br>Age: " + result.age.pred);
          }else{
            console.log(result.gender.prob)
            add_modal_text_and_display($modal, "Image Quality Not Okay! Try Again!")
          }
        }else{
          add_modal_text_and_display($modal, "Please Try Again!")
        }
      },
    });
}

function add_modal_text_and_display($modal, $text){
  $display_msg = $modal.find("#display-msg")
  $display_msg.html($text)
  $modal.modal({backdrop: 'static',keyboard: false})
}

$('.close-btn').click(function(){
  delete_photo_btn.click();
})




