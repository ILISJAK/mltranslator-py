$(document).ready(function(){
    $('#translateBtn').click(function(){
        var text = $('#text').val();
        var src_lang = $('#src_lang').val();
        var tgt_lang = $('#tgt_lang').val();

        if (text.trim() === "") {
            alert("Please enter some text to translate.");
            return;
        }

        $.ajax({
            url: '/translate',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ text: text, src_lang: src_lang, tgt_lang: tgt_lang }),
            success: function(response) {
                $('#translation').text(response.translation);
            },
            error: function() {
                $('#translation').text("Error in translation. Please try again.");
            }
        });
    });
});
