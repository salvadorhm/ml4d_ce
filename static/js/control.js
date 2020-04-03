$(document).ready(function() {
    $('#dataframe').DataTable( {
        "scrollY": true,
        "scrollX": true,
        "lengthMenu": [[15, 25, 50, -1], [15, 25, 50, "All"]]
    } );
} );
